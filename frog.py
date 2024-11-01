def a():
    print(112)

class A:
    def __init__(self):
        self.a = a

    def b(self):
        a()

def main():
    a = A()
    a.b()

if __name__ == '__main__':
    main()