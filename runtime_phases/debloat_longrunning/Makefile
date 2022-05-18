CONFIG_MODULE_SIG=n

obj-m := mykernel.o

CURRENT_PATH:=$(shell pwd)
LINUX_KERNEL:=$(shell uname -r)
LINUX_KERNEL_PATH:=/usr/src/linux-$(LINUX_KERNEL)

all:
	make -C $(LINUX_KERNEL_PATH) M=$(CURRENT_PATH) modules
clean:  
	rm -f *.ko *.o *.mod.o *.mod.c .*.cmd *.symvers  modul*