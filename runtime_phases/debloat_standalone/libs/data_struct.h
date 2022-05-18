/*==========================================================
 * Author:          HTmonster 
 * Description:     Record the main data structure
  ==========================================================*/ 
#ifndef __DATA_STRUCT__
#define __DATA_STRUCT__


#include <linux/types.h>	/* basic types						*/
#include <linux/string.h>	/* string related operations		*/
#include <linux/slab.h>		/* kmalloc							*/
#include <linux/vmalloc.h>

#define MAX_APP_LIST_NUM  (5)      
#define MAX_MONITOR_NUM   (10)
#define MAX_LIB_PAGE_NUM  (512) 
#define MAX_LIB_NAME      (64)
#define MAX_RELY_LIBS     (5)



/////////////////////////////struct//////////////////////////////////////

struct _lib_e_t{
	// mappings from VMA
	char name[MAX_LIB_NAME];				     /* the lib/app name, without path	*/
	ulong start;			
	ulong len;

	char *page_permissions_map;                  /* a map to page permission */
};
typedef struct _lib_e_t lib_element_t;

struct _process_t{ 
	char name[128];
	volatile char rst;        /* have any predictions?*/
	char flag_ld_munmaped;    /* flag to describe is ld munmaped*/  
	ulong procid;

	lib_element_t table[MAX_RELY_LIBS];
	unsigned int ele_num;
};
typedef struct _process_t monitor_process_t;


///////////////////////global variable//////////////////////////////////////
extern char* monitor_app_list[MAX_APP_LIST_NUM];             /* app list to monitor */
extern monitor_process_t monitor_processes[MAX_MONITOR_NUM]; /* process list which are being monitored */

/////////////////////////////////////////Function Define////////////////////////////////////////////
/* app */
int is_monitor_app(const char* name);                       /* Determine if the app is to be monitored */

/*process */
monitor_process_t* start_monitor_process(const char *name, ulong procid);      /* Start monitoring a process */
static inline monitor_process_t* find_free_slot(void);                         /* Find a free process monitor item */
static inline monitor_process_t* get_process_slot(ulong procid);               /* Find the process monitor entry by the process ID */
static inline void free_process_slot(monitor_process_t *ptr);                  /* Release the process monitor entry */
static inline void free_all_processes(void);                                   /* Release all process monitor entry */
static inline ulong is_monitor_process_by_id(ulong id);                        /* Determine if the process being monitored is by ID*/
static inline void init_process_slot(monitor_process_t *ptr,const char* name,ulong procid);  /* Initialize a monitor project */
static inline lib_element_t* find_lib_element_by_name(monitor_process_t *ptr,char* name);  /* Find the lib element in monitor_process_t by lib name */

////////////////////////////////////////function implementations///////////////////////////////////

/**
 * Determine if the application is to be monitored
 */
int is_monitor_app(const char* name){
    ulong i=0;
    for(i=0;i<MAX_APP_LIST_NUM;i++){
        if(monitor_app_list[i]==NULL) break;
        /* matched */
        if(!strcmp(name,monitor_app_list[i])) return 1;
        /* if name to long >15 matched */
        if(strlen(name)>=15 && !strncmp(name,monitor_app_list[i],strlen(name))) return 1;
    }
	return 0;/* not matched */
}



/**
 * find a free location from monitor_processes
 */
static inline monitor_process_t* find_free_slot(void){
	int i =0;
	for(i=0;i<MAX_MONITOR_NUM;i++){
		/* find  */
		if(monitor_processes[i].procid == 0){
			return &monitor_processes[i];
		}
	}
	return NULL;/*not finded*/
}


/**
 * init a slot in the monitor_processes
 */
static inline void init_process_slot(monitor_process_t *ptr,const char* name,ulong procid){
	ptr->procid=procid; /*process id */
	ptr->rst=0;
	ptr->flag_ld_munmaped=0;
	ptr->ele_num=0;

	strcpy(ptr->name,name); /* app name whitout path */
	memset(ptr->table,0,sizeof(lib_element_t)*MAX_RELY_LIBS);           /* init with 0*/
}

/**
* get a process slot by process id
**/
static inline monitor_process_t* get_process_slot(ulong procid){
    int i;
    for(i=0;i<MAX_MONITOR_NUM;i++){
        /*finded*/
        if(monitor_processes[i].procid==procid){
            return &monitor_processes[i];
        }
    }
    /*not finded*/
    return NULL;
}

/**
* free one process
*/
static inline void free_process_slot(monitor_process_t *ptr){
	int i = 0;
	printk(KERN_DEBUG,"*free slot %s\n",ptr->name);
	for(i = 0; i < ptr->ele_num; ++i){
		if(ptr->table[i].page_permissions_map){
			vfree(ptr->table[i].page_permissions_map);
		}
		
	}
	memset(ptr, 0, sizeof(monitor_process_t));
}


/**
* free monitor process list
*/
static inline void free_all_processes(void){
	int i = 0;
	for(i = 0; i < MAX_MONITOR_NUM; i ++){
		if(monitor_processes[i].procid != 0){
			free_process_slot(&monitor_processes[i]);
		}
	}
}


/**
* is a process monitied?
*/
static inline ulong is_monitor_process_by_id(ulong id){
	return !!(ulong)get_process_slot(id);
}

/**
 * start to monitor a process
 */
monitor_process_t* start_monitor_process(const char *name, ulong procid){
	monitor_process_t * ptr = find_free_slot();
	if(ptr == NULL){
		printk(KERN_ERR "[X] No more location to allocate process item");
		return NULL;
	}

	printk(KERN_NOTICE "[+] monitored app [%s], procid %08ld\n", name, procid);
	init_process_slot(ptr, name, procid);
	return ptr;
}

/**
* find the match lib element in monitor_process_t by lib name
*/
static inline lib_element_t* find_lib_element_by_name(monitor_process_t *ptr,char* name){

	int i=0;
	for(i=0;i<ptr->ele_num;i++){
		// printk("[%s] vs [%s]\n",name,ptr->table[i].name);
		if(0==strcmp(name,ptr->table[i].name)){
			return &(ptr->table[i]);
		}
	}

	return NULL;

}

#endif
