
import os, time
import numpy as np
import shmarray as sharedmem


numtypes = [np.float64, np.int32, np.float32, np.uint8, np.complex]

def test_shared_ones():
    for typestr in numtypes:
        shape = (10,)
        a = sharedmem.ones(shape,dtype=typestr)
        t = (a == np.ones(shape))
        assert t.all()


def test_shared_zeros():
    """test sharedmem.zeros for small single axis types"""
    for typestr in numtypes:
        shape = (10,)
        a = sharedmem.zeros(shape,dtype=typestr)
        t = (a == np.zeros(shape))
        assert t.all()

def test_KiB_shared_zeros():
    """test sharedmem.zeros for arrays on the order of 2**16, single axis types"""
    for typestr in numtypes:
        shape = (2**16,)
        a = sharedmem.zeros(shape,dtype=typestr)
        t = (a == np.zeros(shape))
        assert t.all()

def test_MiB_shared_zeros():
    """test sharedmem.zeros for arrays on the order 2**21 bytyes, single axis uint8"""
    
    shape = (2**21,)
    a = sharedmem.zeros(shape,dtype='uint8')
    t = (a == np.zeros(shape))
    assert t.all()


        
import multiprocessing, os, pickle

def test_two_subprocesses_no_pickle():
    #setup
    shape = (4,)
    a = sharedmem.zeros(shape, dtype='float64')
    a = sharedmem.zeros(shape)
    print(os.getpid(),":", a)


    lck = multiprocessing.Lock()

    def modify_array(a,lck):
        # a = pickle.loads(a)
        with lck: #lck.acquire()
        
            a[0] = 1
            a[1] = 2
            a[2] = 3
            # lck.release()
        print(os.getpid(), "modified array")
        
    p = multiprocessing.Process(target=modify_array, args=(a,lck))
    p.start()

    # poll for the result super inefficent!
    t0 = time.time()
    t1 = t0+10
    nn = 0
    while True:
        if a[0]:
            with lck: #             lck.acquire()
                t = (a == np.array([1,2,3,0], dtype='float64'))
                # lck.release()
            break
        
        if time.time() > t1 : # use timeout instead
            break
        nn += 1
    # this will raise an exception if timeout    
    print(os.getpid(), t)
    assert t.all()
    print("finished (from %s)" % os.getpid())
    
    p.join()
    print(a)


def test_two_subprocesses_with_pickle():
    from nose import SkipTest
    raise SkipTest("this test is known to fail")

    shape = (4,)
    a = sharedmem.zeros(shape, dtype='float64')
    a = sharedmem.zeros(shape)
    print(os.getpid(),":", a)
    pa = pickle.dumps(a)

    lck = multiprocessing.Lock()

    def modify_array(pa,lck):
        a = pickle.loads(pa)
        with lck:
            a[0] = 1
            a[1] = 2
            a[2] = 3

        print(os.getpid(), "modified array")
        
    p = multiprocessing.Process(target=modify_array, args=(pa,lck))
    p.start()

    t0 = time.time()
    t1 = t0+10
    nn = 0
    while True:
        if a[0]:
            with lck:
                t = (a == np.array([1,2,3,0], dtype='float64'))
            break
        if time.time() > t1 : # use timeout instead
            break
        nn += 1
        
    print(os.getpid(), t, "nn:", nn)
    assert t.all()
    print("finished (from %s)" % os.getpid())
    
    p.join()
    
    print(a)


def determine_alloc_limit():
        
    def alloc_n(n):
        shape = (2**n,)
        a = sharedmem.zeros(shape,dtype='uint8')
        t = (a == np.zeros(shape))
        assert t.all()

    for n in range(30):
        alloc_n(n)
        print("2**%d succeeded" %n)
