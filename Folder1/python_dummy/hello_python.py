print("Hello Hola python!")
print("Hello Hola python2!")
print("Hello Hola python3!")
message= 'Holi en español' 
print (message)


print(' \" > Imprime comillas')
print(" \\n > Separa texto en una nueva linea ")
print(" \\t > Imprime un tabulador ")
print(" \\ > Imprime la barra invertida textualmente ")


input ("Dime tu nombre: ") #ejecuta de dentro a fuera
print (input ("Dime tu apellido: ")) #imprime el input
print("tu nombre es: " + input ("Dime tu apellido: ") +" "  + input ("Dime tu nombre: "))

name='soy todo lowercase'
print(name.title()) 
print(name.upper()) 
print(name.lower()) 


first_name='Corla'
second_name='Lopez'
full_name= f'{first_name} {second_name}' # f de formato  para poner los valores de las variables {}
print(full_name)
print(f"Holi holi, {full_name.title()} !")
