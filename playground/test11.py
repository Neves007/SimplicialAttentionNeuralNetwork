from multipledispatch import dispatch

@dispatch()
def func():
    print("Default implementation:")

@dispatch(str)
def func(arg):
    print("Implementation for integers:", arg)

@dispatch(float)
def func(arg):
    print("Implementation for strings:", arg)

func()      # Calls the integer implementation
func('hello') # Calls the string implementation
func(2.5)     # Calls the default implementation