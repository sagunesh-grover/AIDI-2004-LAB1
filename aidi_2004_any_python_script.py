#python script to find whether a number is odd or even


num = int(input("Please enter a number: "))
if (num % 2) == 0:
   print("{0} is Even".format(num))
else:
   print("{0} is Odd".format(num))