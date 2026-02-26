import this

long_number=14_000_000
print(long_number)

var1,var2,var3 =1,2,3
print(str(var1) +' '+var2.__str__()+ ' ' + str(var3))


print(f"{var1}   {var2} espacio {var3}")

CONSTANTE='no vario' #esto es una variable pero al dejarla en mayúsculas indico que la trato como constante
print(CONSTANTE)

#listas van con []
bikes=['trek','redline fast','specialized']
print(bikes)
print(bikes[1]) #las posicones empiezan por 0
print(bikes[1].title())
print(bikes[-1]) #te devuelve siempre la ultima posición
print(bikes[-2]) #te devuelve siempre la penúltima posición
print(bikes[-3]) #te devuelve siempre la ante penúltima posición

#valores individuales
message=f"My first bike was a {bikes[0].title()}"
print(message)

message2="My first bike was a "+ bikes[0].title()
print(message2)

print("My first bike was a "+ bikes[0].title())

#MAL >>bikes[3]='La nueva molona'
bikes.append('La nueva molona')
print(bikes)
