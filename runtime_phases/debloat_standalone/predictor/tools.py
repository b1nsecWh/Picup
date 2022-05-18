import time
from functools import wraps

def timer(fn):
    @wraps(fn)
    def measure_time(*args,**kwargs):
        s=time.time()
        rst=fn(*args,**kwargs)
        e=time.time()
        print("[{}] {:.6f}s".format(fn.__name__,e-s))

        return rst
    
    return measure_time

@timer
def function_needs_to_be_measur():
    for i in range(10000):
        #print(i)
        pass

if __name__ =='__main__':
    function_needs_to_be_measur()