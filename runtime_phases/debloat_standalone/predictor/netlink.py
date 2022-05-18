
NETLINK_TEST  =  30

import socket               
import os
from struct import pack,unpack


# flag information
FLAGS={
    'fail'  :0,
    'sucess':1, # 弃用
    'update':2,
}

# struct nlmsghdr
# {
#     __u32 nlmsg_len;   /* Length of message */
#     __u16 nlmsg_type;  /* Message type*/
#     __u16 nlmsg_flags; /* Additional flags */
#     __u32 nlmsg_seq;   /* Sequence number */
#     __u32 nlmsg_pid;   /* Sending process PID */
# };

def netlink_init():
    ''' creat netlink socket '''
    netlink_socket = socket.socket(socket.AF_NETLINK, socket.SOCK_RAW, NETLINK_TEST )
    #netlink_socket.setsockopt(270, 1, 31)
    netlink_socket.bind((os.getpid(), 0))                                            

    netlink_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)            
    netlink_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)

    return netlink_socket

def report_pid_to_kernel(netlink_socket):
    ''' send process pid '''
    data=pack("=IHHII",16,0,FLAGS['update'],0,os.getpid())
    netlink_socket.sendto(data, (0, 0))

def send_fail_signal(netlink_socket):
    ''' send signal to kernel '''
    data=pack("=IHHII",16,0,FLAGS['fail'],0,os.getpid())
    netlink_socket.sendto(data, (0, 0))

def netlink_close(netlink_socket):
    netlink_socket.close()