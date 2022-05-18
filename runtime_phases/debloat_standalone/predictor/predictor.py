import os
import json
import time
from struct import pack,unpack

import torch
import models.channelCNN
import numpy as np
import timeout_decorator



import db
from netlink import *
import tools

device=torch.device('cpu')

MAX_LIB_PAGES=512

models={}
embedding_dict={}
app_pages={}
app_apis={}
vocab_size={}
rely_libs={}

page_map_based=[b'0',]*MAX_LIB_PAGES

LIB_MAX_EXE_PAGES={
    "libc-2.27.so":487,
    "libz.so.1.2.11":28,
    'libbfd-2.30-system.so':297,
    'libdl-2.27.so':3,
    'libopcodes-2.30-system.so':111
}

CUR_PATH=os.path.dirname(os.path.realpath(__file__))

# @tools.timer
def parse_data(raw_data):
    data={}
    raw_data=raw_data.replace(b'\x00',b'')
    raw_data=raw_data.decode("utf-8")
    # print(raw_data)
    for element in raw_data.split(','):
        
        element=element.strip()
        element=element.split(':')
        data[element[0]]=element[1]
    print(data)
    return data


def load_all_models(path=CUR_PATH+"/models"):

    for file in os.listdir(path):
        if file.endswith(".pkl"):
            models[file[:-4]]=torch.load(path+"/"+file,map_location='cpu')
            models[file[:-4]].eval()
    print(models)


def load_app_api_info():

    with open("./app_info.json",'r')as f:
        json_str=f.read()
        app_infos=json.loads(json_str)
        
        for app,info in app_infos.items():
            embedding_dict[app]={}
            for i,word in enumerate(info['params'].split(',')):
                embedding_dict[app][word]=i+1
            vocab_size[app]=i
            app_apis[app]=info['apis'].split(',')
            apidb=db.binaryApi2soDB("./db/cat.db", app)
            
            app_pages[app]=[]
            rely_libs[app]=set()
            for api in app_apis[app]:
                
                querys=apidb.search_by_name(api)
                for query in querys:
                    rely_libs[app].add(query[1].split('/')[-1])
                    app_pages[app].append([eval(query[2]),query[1].split('/')[-1]])


@timeout_decorator.timeout(1)
@tools.timer
def get_input_tensor(app,argv):
        text_embed_size=15
        file_encode_w=32
        text_min_len=4

        input_text,file=argv.split()

        def read_file_to_list(filename):
            with open(filename, "rb") as f:
                content=f.read(file_encode_w*file_encode_w)

                content=list(content[:file_encode_w * file_encode_w])
                if len(content)<(file_encode_w * file_encode_w):
                    content+=[0 for _ in range((file_encode_w * file_encode_w)-len(content))]

                content=np.reshape(np.array(content), (1, file_encode_w, file_encode_w))

                return content
        words=[]
        words+=input_text.split()
        word_embed=[]
        for w in words:
            if w in embedding_dict[app].keys():
                word_embed+=[embedding_dict[app][w] for _ in range(1)]
            else:
                vocab_size[app]+=1
                embedding_dict[app][w]=vocab_size[app]
                word_embed+=[vocab_size[app]  for _ in range(text_embed_size)]

        if len(word_embed) <text_min_len: word_embed+=[0 for __ in range(text_min_len-len(word_embed))]
        word_embed=torch.tensor([word_embed])
        
        if os.path.exists(file):
            file_embed=read_file_to_list(file)
        else:
            raise FileNotFoundError()
        file_embed=torch.tensor([file_embed],dtype=torch.float32)

        return (word_embed.to(device),file_embed.to(device))

@timeout_decorator.timeout(5)
@tools.timer
def model_predictor(app,input_tensor,threshold=0.5):
    
    # 预测
    s=time.time()
    output=model(input_tensor)
    e=time.time()
    print("prediction: ",(e-s)*1000)
    print(len(output[0]))

    data={}
    for lib in rely_libs[app]:
        data[lib]=page_map_based[:LIB_MAX_EXE_PAGES[lib]]

    for i,value in enumerate(output[0]):
        if value > threshold:
            pages,lib=app_pages[app][i]
            for page in pages:
                data[lib][page]=b'1'
    return data

@tools.timer
def send_sucess_result(pid,data,netlink_socket):
    
    print(int(pid.strip()))
    b_data=pack('Q',int(pid.strip()))
    
    for lib,pages in data.items():
        pages=b''.join(pages)
        b_data+=pack(str(len(lib)+1)+'s',lib.encode('utf-8')+b'\0')
        b_data+=pack('1s','@'.encode('utf-8'))
        b_data+=pack('{}s'.format(len(pages)+1),pages+b'\0')
        b_data+=pack('1s','#'.encode('utf-8'))

    b_head=pack("=IHHII",16+len(b_data),0,FLAGS['sucess'],0,os.getpid())
    netlink_socket.sendto(b_head+b_data, (0, 0))
    print("==>",b_head+b_data)
    

def send_fail_signal(pid,netlink_socket):

    print(int(pid.strip()))
    b_data=pack('Q',int(pid.strip()))

    b_head=pack("=IHHII",16+len(b_data),0,FLAGS['fail'],0,os.getpid())

    netlink_socket.sendto(b_head+b_data, (0, 0))
    print("==>",b_head+b_data)



if __name__ == "__main__":

    netlink_socket=netlink_init()

    report_pid_to_kernel(netlink_socket)

    load_app_api_info()
    load_all_models()
    model=models['cat']

    print("========================[listen]==========================")
    while True:
        recv_data, (nlpid, nlgrps) = netlink_socket.recvfrom(1024)
        
        # Netlink message header (struct nlmsghdr)
        msg_len, msg_type, msg_flags, msg_seq, msg_pid \
            = unpack("=IHHII", recv_data[:16])
        # data
        recv_data = recv_data[16:]

        try:
            s=time.time()
            recv_data = parse_data(recv_data)
            input_tensor=get_input_tensor(recv_data['app'],recv_data['argv'])
            rst=model_predictor(recv_data['app'], input_tensor)
            send_sucess_result(recv_data['pid'], rst,netlink_socket)
            e=time.time()
            print("total",(e-s)*1000)
        except Exception as e:
            print(e)
            send_fail_signal(recv_data['pid'],netlink_socket)              

    netlink_close(netlink_socket)