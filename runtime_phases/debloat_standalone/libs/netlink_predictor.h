#ifndef __NETLINK_PREDICTION_H__
#define __NETLINK_PREDICTION_H__ 

#include <linux/init.h>
#include <linux/module.h>
#include <linux/types.h>
#include <net/sock.h>
#include <linux/netlink.h>
#include <linux/delay.h>
#include <linux/vmalloc.h>

#include "utility.h"
#include "data_struct.h"
 
#define NETLINK_TEST     30

/*The status code passed by predictor*/
#define FLAGS_PID_UPDATE 2
#define FLAGS_SUCESS     1
#define FLAGS_FAIL       0



////////////////////////////variable///////////////////////////////////
extern struct net init_net;
extern struct sock * nlsk=NULL;   /* socket communication with python predictor */
extern int predictor_pid=0;       /* python predictor pid*/

extern monitor_process_t monitor_processes[MAX_MONITOR_NUM]; /* process list which are being monitored */

///////////////////////////////////////////Function Define///////////////////////////////////////////
struct sock * netlink_sock_init(void);                                       /* netlink init */
void netlink_sock_clean(struct sock * nlsk);                                 /* Remove netlink */

unsigned int send_info_to_predictor(ulong pid,char* bin,char *argv);          /* Sends input information to the predictor*/

///////////////////////////////////////////Function implementation///////////////////////////////////////////
/**
* parse_prediction
**/
int parse_prediction(char* umasg,unsigned int size){
    ulong pid=0;
    ulong p_num=0;
    int i;

    /* read pid */
    for(i=0;i<8;i++){
        // printk("%d,%c,%u\n",i,p+i,pid);
        pid=pid|(((*(umasg+i))&0xff)<<(8*i));
    }
    p_num+=8;

    /* find pid slot */
    monitor_process_t* slot=get_process_slot(pid);
    if(NULL!=slot){
        while(p_num<size-1){
            /*====== parse lib name ==========*/
            // lib.name=vmalloc(len+1);
            lib_element_t* lib=find_lib_element_by_name(slot,umasg+p_num);
            if(NULL==lib){
                printk(KERN_ERR"[X] No lib matched for %s",umasg+p_num);
                slot->rst=-1;
                return -1;
            }

            int len=strlen(umasg+p_num);
            
            p_num+=len+1;

            /* check */
            if('@'!=*(umasg+p_num)){
                printk(KERN_ERR"[X] parse prediction fmt ERROR;please check @");
                slot->rst=-1;
                return -1;
            }
            p_num+=1;

            /*======parse prediction=========*/
            len=strlen(umasg+p_num);
            if(len>MAX_LIB_PAGE_NUM){
                printk(KERN_ERR"[X] element num > max page size !!!");
                slot->rst=-1;
                return -1;
            }
            lib->page_permissions_map=vmalloc(len+1);
            strcpy(lib->page_permissions_map,umasg+p_num);
            p_num+=len+1;
            printk(KERN_NOTICE "\t >>rst: %s",lib->page_permissions_map);

            /* check */
            if('#'!=*(umasg+p_num)){
                printk(KERN_ERR"[X] parse prediction fmt ERROR;please check");
                slot->rst=-1;
                return -1;
            }
            p_num+=1;
            
        }
        slot->rst=1;

        return 1;
    
        
    }else{
        printk(KERN_ERR "[X] slot of %u disappear! \n",pid);
        return -1;
    }
}

/**
* parse_error
**/
void parse_error(char* umasg,unsigned int size){
    ulong pid=0;
    ulong p_num=0;

    /* read pid */
    int i;
    for(i=0;i<8;i++){
        // printk("%d,%c,%u\n",i,p+i,pid);
        pid=pid|(((*(umasg+i))&0xff)<<(8*i));
    }

    monitor_process_t* slot=get_process_slot(pid);
    if(NULL!=slot){
        slot->rst=-1;
    }else{
        printk(KERN_ERR "[X] slot of %u disappear! \n",pid);
    }
}
/**
* msg recv callback
*/
static void netlink_rcv_msg(struct sk_buff *skb)
{
    struct nlmsghdr *nlh ;
    nlh= NULL;

    if(skb->len >= nlmsg_total_size(0))
    {
        nlh = nlmsg_hdr(skb);  //Get nlmsghdr from sk_buff.
        int flags;
        flags=nlh->nlmsg_flags;
        printk(KERN_NOTICE "\t<======== netlink recived msg. \t flag: %d\n",flags);
        if(flags==FLAGS_PID_UPDATE){
            /* update pid */
            predictor_pid=nlh->nlmsg_pid;
            printk(KERN_NOTICE "[+] updated predictor pid:%d \n",predictor_pid);
            return 0;
        }else if(flags==FLAGS_SUCESS){
            /* read prediction */
            printk(KERN_NOTICE "----------[reading prediction len:%d ]--------------\n",nlh->nlmsg_len);
            char* umsg = nlmsg_data(nlh); //Get payload from nlmsghdr.
            parse_prediction(umsg,(nlh->nlmsg_len)-16);
            printk(KERN_NOTICE "------------------------------------------------------\n",nlh->nlmsg_len);
        }else if(flags==FLAGS_FAIL){
            printk(KERN_ERR "---------------[prediction error ]--------------\n",nlh->nlmsg_len);
            char* umsg = nlmsg_data(nlh); //Get payload from nlmsghdr.
            parse_error(umsg,(nlh->nlmsg_len)-16);
            printk(KERN_NOTICE "------------------------------------------------------\n",nlh->nlmsg_len);


        }
        
    }
}

/**
* send the argument to python client
**/
int send_params_to_predictor(char* arguments){
    struct sk_buff *nl_skb;
    struct nlmsghdr *nlh;
    int ret;
    uint16_t len;

    printk(KERN_NOTICE "\t========> send argument [%s] to predictor\n",arguments);
    
    len=strlen(arguments);
    nl_skb = nlmsg_new(len, GFP_ATOMIC);
    if(!nl_skb){
        printk(KERN_ERR "[X] netlink alloc failure\n");
        return -1;
    }
    nlh = nlmsg_put(nl_skb, 0, 0, NETLINK_TEST, len, 0);
    if(nlh == NULL){
        printk(KERN_ERR "[X] nlmsg_put failaure \n");
        nlmsg_free(nl_skb); 
        return -1;
    }
    memcpy(nlmsg_data(nlh),arguments, len);
    ret = netlink_unicast(nlsk, nl_skb, predictor_pid, 0);
    if (ret < 0){
        printk(KERN_ERR "\t\t error: %d\n", ret);
    }
    return ret;
}

/**
*creat sock
*/
struct sock * netlink_sock_init(void)
{   
    struct sock *nlsk = NULL;

    struct netlink_kernel_cfg cfg = { 
        .input  = netlink_rcv_msg, /* set recv callback */
    };  

    /* Create a Netlink socket */
    nlsk = (struct sock *)netlink_kernel_create(&init_net, NETLINK_TEST, &cfg);
    if(nlsk == NULL)
    {   
        printk(KERN_ERR "[X] netlink_kernel_create error !\n");
        return NULL; 
    }   
    printk(KERN_NOTICE "[+] socket create successful\n");
    
    return nlsk;
}

/**
* clean the netlink
*/
void netlink_sock_clean(struct sock * nlsk)
{   
    if (nlsk){
        netlink_kernel_release(nlsk); /* release ..*/
        nlsk = NULL;
    }   
    printk(KERN_NOTICE "[-] socket relese successful\n");
}

/**
* send info to predictor
*/
unsigned int send_info_to_predictor(ulong pid,char* bin,char *argv){
    int ret;
    char  arguments[512]	=	{0};

    sprintf(arguments, "pid:%ld,app:%s,argv:%s",pid,bin,argv);
    ret=send_params_to_predictor(arguments);

    return ret;
}




#endif // !__NETLINK_PREDICTION_H__