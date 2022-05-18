#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/kprobes.h>
#include <linux/init.h>
#include <linux/mman.h>		/* mmap2, PROT_EXEC			*/
#include <linux/file.h>		/* fget						*/

#include "libs/data_struct.h"
#include "libs/utility.h"
#include "libs/netlink_predictor.h"

/////////////////////////////////////////////////////////////
char* monitor_app_list[MAX_APP_LIST_NUM];            /* app list to monitor */
monitor_process_t monitor_processes[MAX_MONITOR_NUM];/* process list */

/*sys_mprotect*/
long (*orig_sys_mprotect)(unsigned long start, size_t len,unsigned long prot);


/*=================================================[init]=============================================*/
/*init app list to monitor*/
void init_monitor_app_list(void){
    monitor_app_list[0]="/usr/sbin/nginx";
}

/*init netlink */
int sock_init(void){
	nlsk=netlink_sock_init();
	if(NULL==nlsk) printk(KERN_ERR "[x] sock init error!!");
	return 0;
}
/*----------------------- sys_execve ---------------------------- */

int jsys_execve(const char __user *filename,
	const char __user *const __user *__argv,
	const char __user *const __user *__envp)
{	
  	if(is_monitor_app(filename)){
		char* short_name 		= 	get_filename_from_path(filename);	/*app name (short)*/
		ulong procid			=	current->pid;                  		/*process id*/

		monitor_process_t* slot=start_monitor_process(short_name,procid);        /* 1. Create monitoring slot */
		printk(KERN_INFO "[*] execve name: %s pid:%ld\n",short_name,procid);
  	}
  	jprobe_return();
  	return 0;
 }


/*----------------------- sys_mmap ---------------------------- */

/* mmap hook data */
struct mmap_kret_data {
	ulong  addr;
	ulong  len;
	ulong  prot;
	ulong  fd;
	off_t off;   
	int is_monitor;
};

/* mmap entry */
static int mmap_entry_handler(struct kretprobe_instance *ri, struct pt_regs *regs){
	struct mmap_kret_data *data;
	data = (struct mmap_kret_data *)ri->data;

	/* skip kernel thread */
	// if(!current->mm){return 1;}

	if(is_monitor_process_by_id(current->pid)){
		data->is_monitor=1;
		/*record params*/

		data->addr=	(ulong)(regs->di);
		data->len = (ulong)(regs->si);
		data->prot= (ulong)(regs->dx);
		data->fd  = (ulong)(regs->r8);
		data->off = (off_t)(regs->r9);
	}else{
		data->is_monitor=0;
	}
	return 0;
}

/* mmap ret */
static int mmap_ret_handler(struct kretprobe_instance *ri, struct pt_regs *regs)
{
	ulong retval = regs_return_value(regs);
	struct mmap_kret_data *data = (struct mmap_kret_data *)ri->data;

	if(data->is_monitor && (data->prot & PROT_EXEC) != 0 && (((long)(data->fd) > 0))){
		struct file *fil = fget(data->fd);
		char* name = fil->f_path.dentry->d_iname;

		printk(KERN_INFO "[*] mmap %s pid:%u addr:%u len: %u off:%u ret=> %u\n",name,current->pid,data->addr,data->len,data->off,retval);
		
		monitor_process_t* slot=get_process_slot(current->pid);

		if(NULL != slot){
			//set lib mmaped information
			lib_element_t* lib=&(slot->table[slot->ele_num]);
			strcpy(lib->name,name);
			lib->start=retval;
			lib->len=data->len;

			slot->ele_num++;
		}else{
			printk(KERN_ERR "[X] No slot of {}",current->pid); 
		}
	}
	return 0;
}

/*------------------------------------------------[sys_input_hook]----------------------------------------*/


struct recvfrom_kret_data {
	int fd ;
	void* ubuf;
	size_t size;
	unsigned int flags;
};

/*
For long-running process, like Nginx, we debloat libraries after each input is obtained.
*/
static int recvfrom_entry_handler(struct kretprobe_instance *ri, struct pt_regs *regs){

	struct recvfrom_kret_data* data;
	data = (struct recvfrom_kret_data* )ri->data;

	if(is_monitor_process_by_id(current->pid)){
		data->fd		=(int)(regs->di);
		data->ubuf 		=(void*)(regs->si);
		data->size		=(size_t)(regs->dx);
		data->flags		=(unsigned int)(regs->cx);
	}
	return 0;
}


static int recvfrom_ret_handler(struct kretprobe_instance *ri, struct pt_regs *regs)
{
	struct recvfrom_kret_data* data = (struct recvfrom_kret_data* )ri->data;
	ulong retval = regs_return_value(regs);

	if(retval!=0 && is_monitor_process_by_id(current->pid)){

		printk("%s\n",(char*)data->ubuf);
		
		monitor_process_t* slot=get_process_slot(current->pid);
		if(NULL != slot){
			/*1. send input to predictor*/
			send_info_to_predictor(current->pid,slot->name,(char*)data->ubuf);
			/*2. wait for prediction result */
			ulong k=0;
			while(0==slot->rst&&k<10000000000){k++;}
			if(slot->rst!=1){
				printk(KERN_ERR "[X] No predition result skip permission set");
				return 0;
			}
			/*3. process prediction result*/
			int i,all_pages,exe_pages;
			for(i=0;i<slot->ele_num;i++){
				exe_pages=0;
				lib_element_t* lib=&(slot->table[i]);

				if(NULL==lib->page_permissions_map){
					printk(KERN_ERR "[X] NULL; No prediction result for %s, skip permission set",lib->name);
					continue;
				}
				all_pages=strlen(lib->page_permissions_map);

				/*step 1: Set all the executable page is No executable*/
				orig_sys_mprotect(lib->start,PAGE_SIZE*all_pages,PROT_READ);
				/*step 2: Set the memory pages that can be run*/
				int j;
				for(j=0;j<all_pages;j++){
					if('1'==lib->page_permissions_map[j]){
						exe_pages++;
						orig_sys_mprotect(lib->start+(PAGE_SIZE*j),PAGE_SIZE,PROT_READ|PROT_EXEC);
					}
				}

				printk("[*] enable protection on: %s	=> all_pages:%d  exe_pages:%d\n",lib->name,all_pages,exe_pages);
			}

		}else{
			printk(KERN_ERR "[X] No slot of {}",current->pid); 
		}
	}
	return 0;
}



/*--------------------------------------------sys_clone---------------------------------------------*/

/* mmap ret */
static int clone_ret_handler(struct kretprobe_instance *ri, struct pt_regs *regs)
{
	ulong retval = regs_return_value(regs);
	
	if(is_monitor_process_by_id(current->pid)){
		printk(KERN_INFO "[*] clone current:%d clone:%d",current->pid,retval);

		monitor_process_t* slot=get_process_slot(current->pid);

		if(NULL != slot){
			slot->procid=retval;
		}
	}
	return 0;
}


// /*----------------------- sys_exit ---------------------------- */
void jdo_exit(long code){
	ulong procid = current->pid;
	monitor_process_t *slot = get_process_slot(procid);

	if(slot != NULL){
		free_process_slot(slot);             /* release the pid slot */
		printk("[-] exit %u\n",procid);
	}
	jprobe_return();
	return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static struct jprobe jprobe_execve   = {  .entry = jsys_execve, .kp = {.symbol_name	= "sys_execve",},};
/* sys_mmap [kretprobes] */
static struct kretprobe mmap_kretprobe = { 
	.handler = mmap_ret_handler,                .entry_handler = mmap_entry_handler,
 	.data_size = sizeof(struct mmap_kret_data), .maxactive	= 1,
 	.kp = { .symbol_name	= "sys_mmap_pgoff",},                                   };

/* sys_clone [kretprobes] */
static struct kretprobe clone_kretprobe = { 
	.handler = clone_ret_handler,                   
 	.kp = { .symbol_name	= "sys_clone",},                                   };

static struct kretprobe recvfrom_kretprobe = { 
	.handler = recvfrom_ret_handler,                
	.entry_handler = recvfrom_entry_handler,
 	.data_size = sizeof(struct recvfrom_kret_data), 
 	.kp = { .symbol_name	= "sys_recvfrom",},};


// // /* do_exit  [jprobes]  */
static struct jprobe jprobe_exit     = {  .entry = jdo_exit,    .kp = {.symbol_name	= "do_group_exit",   },};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int probes_init(void)
{
	int ret;

	/* jprobe_execve hook */
	ret = register_jprobe(&jprobe_execve);
	if (ret < 0) {
		pr_info("register_jprobe  jprobe_execve  failed, returned %d\n", ret);
		return -1;
	}
	pr_info("Planted jprobe execve at %p, handler addr %p\n",jprobe_execve.kp.addr, jprobe_execve.entry);

	/* mmap_kretprobe hook */
	ret = register_kretprobe(&mmap_kretprobe);
	if (ret < 0) {
		printk(KERN_INFO "register_kretprobe failed, returned %d\n",ret);
		return -1;
	}

	printk(KERN_INFO "Planted return probe at %s: %p\n",
			mmap_kretprobe.kp.symbol_name, mmap_kretprobe.kp.addr);

	/* mmap_kretprobe hook */
	ret = register_kretprobe(&clone_kretprobe);
	if (ret < 0) {
		printk(KERN_INFO "register_kretprobe failed, returned %d\n",ret);
		return -1;
	}

	printk(KERN_INFO "Planted return probe at %s: %p\n",
			clone_kretprobe.kp.symbol_name, clone_kretprobe.kp.addr);

	ret = register_kretprobe(&recvfrom_kretprobe);
	if (ret < 0) {
		printk(KERN_INFO "register_kretprobe failed, returned %d\n",ret);
		return -1;
	}

	printk(KERN_INFO "Planted return probe at %s: %p\n",
			recvfrom_kretprobe.kp.symbol_name, recvfrom_kretprobe.kp.addr);
	

	ret = register_jprobe(&jprobe_exit);
	if (ret < 0) {
		pr_info("register_jprobe  jprobe_exit  failed, returned %d\n", ret);
		return -1;
	}
	pr_info("Planted jprobe_exit  at %p, handler addr %p\n",jprobe_exit.kp.addr, jprobe_exit.entry);

	return 0;
}


void probes_exit(void){
	unregister_jprobe(&jprobe_execve);
	unregister_kretprobe(&mmap_kretprobe);
	unregister_kretprobe(&clone_kretprobe);
	unregister_kretprobe(&recvfrom_kretprobe);
	unregister_jprobe(&jprobe_exit);
}




///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static int __init mymodule_init(void){
	init_monitor_app_list();
	probes_init();
	sock_init();

	/* get mprotect function */
	orig_sys_mprotect=(void*)kallsyms_lookup_name("sys_mprotect");
	if(orig_sys_mprotect==NULL){
		printk(KERN_ERR "[x]can't find sys_mprotect");
		return -1;
	}
	return 0;
}

static void __exit mymodule_exit(void){	
	//1. Unload the probe
	probes_exit();
	//2. Clear data
	free_all_processes();
	//3. Close netlink
	netlink_sock_clean(nlsk);
}



////////////////////////////////////////////////////////
module_init(mymodule_init)
module_exit(mymodule_exit)
MODULE_LICENSE("GPL");