/*==========================================================
 * Author:          HTmonster 
 * Description:     Utilities Utility Functions Utils
  ==========================================================*/ 

#ifndef __UTILITY_H__
#define __UTILITY_H__

#include <linux/fs.h>
#include <asm/segment.h>
#include <asm/uaccess.h>
#include <linux/slab.h>
#include <linux/buffer_head.h>
#include <linux/vmalloc.h>

////////////////////////////////////extern function///////////////////////////////////////////
extern char* get_filename_from_path(const char* path);
extern char* parse_argv(const char __user *const __user *__argv);

//////////////////////////////////function implementations///////////////////////////////////


/**
 * get the filename from the path
 **/
char* get_filename_from_path(const char* path)
{
	return (char*)(strrchr(path, '/') ? strrchr(path, '/') + 1 : path); 
}

/**
 * parse the running app argv
 */
char* parse_argv(const char __user *const __user *__argv){

	char * exec_str = NULL;
	char * temp=NULL;
	char ** p_argv = (char **) __argv;
	long exec_line_size = 1; 
 
	p_argv++;//skip bin name
	while (NULL != *p_argv) {
		printk("%s %d\n",*p_argv,strlen(*p_argv));
		exec_line_size += (strlen(*p_argv)+1);
		p_argv++;	
	}
	
	exec_str = vmalloc(exec_line_size);
	temp	 = vmalloc(exec_line_size);

	if (NULL != exec_str) {
		p_argv = (char **)__argv;
		p_argv++;
		while (NULL != *p_argv) {
			strcat(exec_str,*p_argv);
			strncat(exec_str," ",1);
			p_argv++;	
		}
	}
	exec_str[exec_line_size-1]='\0';

	vfree(temp);

	return exec_str;
}

#endif