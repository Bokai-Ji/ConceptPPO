# distutils: sources = c-algorithms-master\src\queue.c
# distutils: include_dirs = c-algorithms/src/

# file: queue_.pyx
cimport cqueue

cdef class Queue:
    cdef cqueue.Queue* _c_queue

    def __cinit__(self):
        self._c_queue = cqueue.queue_new()
        if self._c_queue is NULL:
            raise MemoryError()

    cpdef append(self, tuple value):
        if not cqueue.queue_push_tail(self._c_queue, <void*> value):
            raise MemoryError()

    cpdef tuple peek(self) except? -1:
        cdef tuple value = <tuple>cqueue.queue_peek_head(self._c_queue)
        if cqueue.queue_is_empty(self._c_queue):
            raise IndexError("Queue is empty")
        return value

    cpdef tuple pop(self) except? -1:
        if cqueue.queue_is_empty(self._c_queue):
            raise IndexError("Queue is empty")
        return <tuple>cqueue.queue_pop_head(self._c_queue)
    
    def __bool__(self):
        return cqueue.queue_is_empty(self._c_queue)

    def __dealloc__(self):
        if self._c_queue is not NULL:
            cqueue.queue_free(self._c_queue)