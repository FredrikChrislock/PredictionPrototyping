import multiprocessing as mp
from multiprocessing import Process

def f(name):
    print('Hello %s' % (name))

if __name__ == '__main__':
    p = Process(target=f, args=('Fredrik',))
    p.start()
    p.join()
    print("Done")

# Test the other way