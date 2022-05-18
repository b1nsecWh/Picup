import os
import json
import re
import time
from struct import pack,unpack


import timeout_decorator

import torch
import numpy as np

from netlink import *
import db
import models.channelCNN



models={}
words_dict={}
app_apis={}
rely_libs={}
app_pages={}

# executable page count
LIB_MAX_EXE_PAGES={
    "libc-2.27.so":487,
    "libz.so.1.2.11":28,
    'libbfd-2.30-system.so':297,
    'libdl-2.27.so':3,
    'libopcodes-2.30-system.so':111,
    'libpthread-2.27.so':26,
    'librt-2.27.so':7,
    'libm-2.27.so':413,
    'libpcre.so.3.13.3':112,
    'libcrypto.so.1.1':1,
    'libssl.so.1.1':129,
    'libcrypt-2.27.so':9,
}
# max pages 
MAX_LIB_PAGES=512
page_map_based=[b'0',]*MAX_LIB_PAGES

CUR_PATH=os.path.dirname(os.path.realpath(__file__))

def parse_data(raw_data):
    ''' parse input data '''
    data={}
    print(raw_data)
    raw_data=raw_data.split(b'\r\n\r\n')[0]
    raw_data=raw_data.replace(b'\x00',b'')
    raw_data=raw_data.decode("utf-8")
    # print(raw_data)
    for element in raw_data.split(','):
        
        element=element.strip()
        element=element.split(':')
        data[element[0]]="".join(element[1:])
    return data

def load_all_models(path=CUR_PATH+"/models"):
    ''' load prediction model '''

    for file in os.listdir(path):
        if file.endswith(".pkl"):
            models[file[:-4]]=torch.load(path+"/"+file,map_location='cpu')
            models[file[:-4]].eval()
    print(models)


def load_app_api_info():
    ''' load api base information and api dependency '''

    with open("./app_info.json",'r')as f:
        app_infos=json.load(f)
        
        for app,info in app_infos.items():
            words_dict[app]=info['words']
            app_apis[app]=info['apis'].split(',')
            apidb=db.binaryApi2soDB("./db/nginx.db", 'nginx')
            
            app_pages[app]=[]
            rely_libs[app]=set()
            for api in app_apis[app]:
                
                querys=apidb.search_by_name(api)
                api_lib_pages=[]
                for query in querys:
                    lib=query[1].split('/')[-1]
                    pages=eval(query[2])

                    api_lib_pages.append((pages,lib))
                    rely_libs[app].add(lib)
                app_pages[app].append(api_lib_pages)
    
    if "ld-2.27.so" in rely_libs['nginx']:
        rely_libs['nginx'].remove('ld-2.27.so')

def get_input_tensor(app,input):
    min_len=4
    words=[]
    for w in input.split():
        if "." in w or "/" :
            if w=='127.0.0.1' or w=="HTTP/1.1" or w=="*/":
                words.append(w)
            elif w.endswith("/"):
                words.append(w)
            else:   
                for i in re.split(r"\.|\/|\=|-",w):
                    words.append(i)
        else:
            if 2<len(w)<10:words.append(w)
    word_embed=[]
    for w in words:
        if w in words_dict[app].keys():
            word_embed.append(words_dict[app][w])
        else:
            #word_embed.append(random.randint(self.word_vocab_size,self.word_vocab_size*2))
            word_embed.append(0)
            
    if len(word_embed) <min_len: 
        word_embed+=[0 for __ in range(min_len-len(word_embed))]
    
    word_embed=torch.tensor([word_embed])

    return word_embed

@timeout_decorator.timeout(5)
def model_predictor(app,input_tensor,threshold=0.6):
    
    # prediction
    s=time.time()
    output=model(input_tensor)
    e=time.time()
    print("prediction: ",(e-s)*1000)
    data={}
    for lib in rely_libs[app]:
        data[lib]=page_map_based[:LIB_MAX_EXE_PAGES[lib]]

    print(len(output[0]))
    for i,value in enumerate(output[0]):
        # record page can be runned.
        if value > threshold:
            lib_pages=app_pages[app][i]
            for (pages,lib) in lib_pages:
                if lib=='ld-2.27.so':
                    continue
                for page in pages:
                    data[lib][page]=b'1'
    return data

def send_sucess_result(pid,data,netlink_socket):
    # pid
    print(int(pid.strip()))
    b_data=pack('Q',int(pid.strip()))
    
    for lib,pages in data.items():
        pages=b''.join(pages)

        # library name
        b_data+=pack(str(len(lib)+1)+'s',lib.encode('utf-8')+b'\0')
        # split flag
        b_data+=pack('1s','@'.encode('utf-8'))
        # executable pages
        b_data+=pack('{}s'.format(len(pages)+1),pages+b'\0')
        # end flag
        b_data+=pack('1s','#'.encode('utf-8'))


    # header
    b_head=pack("=IHHII",16+len(b_data),0,FLAGS['sucess'],0,os.getpid())
    # send data
    netlink_socket.sendto(b_head+b_data, (0, 0))
    print("==>",b_head+b_data)
    

def send_fail_signal(pid,netlink_socket):
    ''' send data '''

    print(int(pid.strip()))
    b_data=pack('Q',int(pid.strip()))

    b_head=pack("=IHHII",16+len(b_data),0,FLAGS['fail'],0,os.getpid())

    netlink_socket.sendto(b_head+b_data, (0, 0))
    print("<==",b_head+b_data)



if __name__ == "__main__":

    # setup netlink
    netlink_socket=netlink_init()

    # report netlink process pid
    report_pid_to_kernel(netlink_socket)

    # load model and information
    load_app_api_info()
    load_all_models()
    model=models['nginx']

    # netlink listening
    print("========================[listen]==========================")
    while True:
        recv_data, (nlpid, nlgrps) = netlink_socket.recvfrom(1024)
        
        # Netlink message header (struct nlmsghdr)
        msg_len, msg_type, msg_flags, msg_seq, msg_pid \
            = unpack("=IHHII", recv_data[:16])
        # data
        recv_data = recv_data[16:]
        
        
        # recived input, then make prediction
        try:
            s=time.time()
            recv_data = parse_data(recv_data)
            print("==>%s"%(recv_data))
            input_tensor=get_input_tensor(recv_data['app'],recv_data['input'])
            rst=model_predictor(recv_data['app'], input_tensor)
            send_sucess_result(recv_data['pid'], rst,netlink_socket)
            e=time.time()
            print("total",(e-s)*1000)
        except Exception as e:
            raise e
            send_fail_signal(recv_data['pid'],netlink_socket)                # fail

    # close
    netlink_close(netlink_socket)