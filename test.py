# 终极版作用域

name = "lzl"


def f1():
    print(name)


def f2():
    name = "eric"
    f1()


f2()

# 输出：lzl