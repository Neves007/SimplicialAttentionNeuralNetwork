def fun(*args,**kwargs):
    print("\nrun fun!")
    print("prin args!",args)
    print("prin kwargs!",kwargs)

fun()
fun(1)
fun(1,{"A":"B"})