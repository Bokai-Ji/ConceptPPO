cdef extern from "c-algorithms-master\src\queue.h":
    ctypedef struct Queue:  #! 与 cdef struct Queue 区分，cdef在C语言中的引用为struct Queue，而ctypedef为Queue
        pass
    ctypedef void* QueueValue
    
    Queue* queue_new()
    void queue_free(Queue* queue)

    int queue_push_head(Queue* queue, QueueValue data)
    QueueValue queue_pop_head(Queue* queue)
    QueueValue queue_peek_head(Queue* queue)

    int queue_push_tail(Queue* queue, QueueValue data)
    QueueValue queue_pop_tail(Queue* queue)
    QueueValue queue_peek_tail(Queue* queue)

    bint queue_is_empty(Queue* queue)   #! bint在C中使用时为普通的int类型，而在Python中映射为bool类型

"""
请注意这些声明与头文件声明几乎完全相同，因此您通常可以将它们复制过来。
但是，您不需要像上面那样提供所有声明，只需要复制在代码或其他声明中使用的声明，这样 Cython 就可以获取足够和完整的子集。
然后，考虑对它们进行一些调整，以使它们在 Cython 中更合适。

最好为使用的每一个库定义一个`.pxd`文件。
Cython中附带了一组标准的`.pxd`文件, 主要是cpython, libc 以及 libcpp
Numpy库还有一个标准的`.pxd`文件 numpy, 因为它经常在Cython代码中使用
完整列表参阅Cython的`Cython/Includes`源包
"""