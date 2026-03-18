edad=input("dime tu edad ")
print(type(edad))

edad =int(edad)
print(type(edad))

new_edad=1+edad
print(new_edad)

edad2=input("dime tu edad ")
edad2=int(edad2)
print(edad2+1)

#fallo print("tu nueva edad es : "+ new_edad )

#dos formas de hacerlo , mejor la primer de format
print(f"tu  edad es : {edad} ")

print("tu nueva edad es : {} y {}".format(new_edad, edad2+1) #forma antigua y confusa si hy mas de 1 cvalor

#se pueden añadir operaciones 
print(f"tu nueva edad es : {edad+10} ")




