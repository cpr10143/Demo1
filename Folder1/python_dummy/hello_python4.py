
print('-------------------*********************************------------------------')
"""
This module demonstrates Python logical operators, control flow, and loops.
The code is organized into three main sections:
1. Logical Operators (Comparison and Boolean Operations):
    - Tests equality comparisons with integers, strings, and floats
    - Demonstrates string case sensitivity and type differences
    - Shows floating-point calculations and logical AND operations
2. Control Flow (if/elif/else statements):
    - Simple conditional branching based on numeric comparison
    - Multi-branch conditional logic for pet type classification
    - Nested conditionals for age and grade validation
    - Complex boolean logic for skill requirement verification
3. Loops (for iteration):
    - Iterates through a list of characters
    - Demonstrates different print formatting methods
    - Shows list indexing within loop iterations
    - Uses f-strings for formatted output with multiple variables
The code serves as educational exercises for learning Python fundamentals
including boolean operators, control structures, and iteration patterns.
"""
print('-------------------****EJER CURSO VIDEO OPE LOGICOS******------------------------')
print('-------------------********************************------------------------')
print('\n')

mibool= 10==25
print(mibool) #False
mibool= 5+5==18-8
print(mibool) #true

mibool= 'blanco'=='BlanCO'
print(mibool) #False
mibool= 'blanco'=='BlanCO'.lower()
print(mibool) #True
mibool= 100=='100' 
print(mibool) #False
mibool= 100.0==100
print(mibool) #True

var1=25**0.5
var2=5
mibool2= var1==var2
print(mibool2) #True

mibool3=4<5 and 5 >6
print(mibool3) #false


print('-------------------*********************************------------------------')
print('-------------------****EJER CURSO VIDEO CONTROL DE FLUJOS******------------------------')
print('-------------------********************************------------------------')
print('\n')

if 10>9:
    print('es correcto')
else:
    print('no es correcto')

mascota='perro'

if mascota=='gato':
    print('tienes un gato')
elif mascota=='perro' :
    print('tienes un perro')
elif mascota=='pez':
    print('tienes un pez')
else: 
    print('No sé que animal tienes')

edad = 16
nota=9

if edad < 18 :
    print('eres menos de edad')
    if nota>=7: 
        print('has aprobado')
    else:
        print('no aprobado')
else:
    print('eres adulto')

python=True
ingles=False
print(type(python))

if python and ingles:
    print('cumplas con los requisitos para postularte')
elif python and not ingles :
    print('Para postularte, necesitas tener conocimientos de inglés')
elif not python and ingles:
    print('Para postularte, necesitas tener conocimientos de python')
#elif not python and not ingles:
else:
    print('Para postularte, necesitas saber programar en Python y tener conocimientos de inglés')



print('-------------------*********************************------------------------')
print('-------------------****EJER CURSO VIDEO LOOPS FOR******------------------------')
print('-------------------********************************------------------------')
print('\n')

lista=['a','b','c']

for letrita in lista:
    num_letra=lista.index(letrita) 
    #para cambiar en todo el codigo el nombre de una variable  BR rename symbol
    print('hola '+letrita)    #hola a
    print('hola ',   letrita) #hola  a
    print(f'hola, la letra {letrita} esta en la posicion {num_letra}')

lista2=['luis','fede','julia','laura']

for nombre in lista2:
    if nombre.startswith('l'):
        print(f'{nombre} empieza por l')
    else:
        print('el nombre no empieza por L')

lista3=[1,2,3]
valor=0
for numero in lista3:
    valor=valor+numero
    print(valor) #6

lista4=[[1,2],[3,4],[5,6]]
for a,b in lista4:
    print(f'a={a} b={b}') #a=[1, 2] b=[3, 4] a=[5, 6]

for var1 in lista4:
    a=var1[0]
    b=var1[1]
    print(f'a={a} b={b}') #a=1 b=2 a=3 b=4 a=5 b=6
    print(var1)    

#recorrer diccionario con for    
dic={'clave1':'a','clave2':'b','clave3':'c'}

for item in dic:
    print(item) #clave1 clave2 clave3
    print(dic[item]) #a b c

for item in dic.items():
    print(item)  #('clave1', 'a') ('clave2', 'b') ('clave3', 'c')

for item in dic.values():
    print(item) #a b c

for a,b in dic.items():
    print(a,b)  #clave1 a clave2 b clave3 c

# realiza la suma de todos los números pares e impares* por separado en las variables suma_pares y suma_impares respectivamente
ejer_1114=[1,5,8,7,6,8,2,5,2,6,4,8,5,9,8,3,5,4,2,5,6,4]
suma_pares=0
suma_impares=0
# suma_pares  suma_impares  // num % 2 == 0 (valores pares) num % 2 == 1 (valores impares)
for num in ejer_1114:
    if num%2 == 0 :
        suma_pares=suma_pares+num
    else:
        suma_impares=suma_impares+num
print(f'la suma de los pares es {suma_pares} y la suma de los impares es {suma_impares}')


lista5=[1,2,3,4,5]
lista6=[6,7,8,9,10]
lista5=lista6
print(lista5) #[6, 7, 8, 9, 10]

print('-------------------*********************************------------------------')
print('-------------------****EJER CURSO VIDEO LOOPS WHILE******------------------------')
print('-------------------********************************------------------------')
print('\n')

monedas=5

while monedas>0 :
    print(f'tengo {monedas} monedas')
    monedas=monedas-1
  #  monedas -=1 #operador de asignación compuesto, hace lo mismo que monedas=monedas-1
else : print('me he quedado sin monedas')

respuesta='s'
while respuesta=='s':
    respuesta= input('¿quieres seguir? (s/n)')
else:
    print('gracias por jugar')


#break >> se produce la salida del bucle.
#continue >> interrumpe la iteración actual dentro delbucle, llevando al programa a la parte superior del bucle.
#pass >> no altera el programa: ocupa un lugar donde se espera una declaración, pero no se desea realizar una acción.

nombre= input('tu nombre: ')
for letra in nombre:
    if letra=='i':
        break 
    print(letra) #con break se imprime hasta paqu (nombre paquito)

nombre= input('tu nombre para continue: ')
print(nombre)
for letra in nombre:
    if letra=='i':
        continue 
    print(letra) #con break se imprime hasta paquto (nombre paquito)

nombre= input('tu nombre para pass: ')
for letra in nombre:
    if letra=='i':
        pass 
    print(letra) #con break se imprime hasta paquito (nombre paquito)



#Un loop For a lo largo de la siguiente lista de números, imprimiendo en pantalla cada elemento, e interrumpe el flujo en el momento que encuentres un valor negativo:
ejer118= [4,5,8,7,6,9,8,2,4,5,7,1,9,5,6,-1,-5,6,-6,-4,-3]
for num in ejer118 :
    if num >=0 :
        print(num)
    else :
        break


print('-------------------*********************************------------------------')
print('-------------------****EJER CURSO VIDEO LOOPS-RANGEEEEE******------------------------')
print('-------------------********************************------------------------')
print('\n')

lista=list(range(1,101))
print(lista)

#EJER---------------------------------------
suma_cuadrados=0
lista2=list(range(1,16))
for num in lista2:
    ejer122=num*2
    suma_cuadrados +=ejer122

#MEJORADO-----------------------------------
suma_cuadrados=0
for num in range(1,16):
    suma_cuadrados += num**2


lista3=list(range(0,10))
print(lista3) #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(sum(lista3)) #45
print(max(lista3)) #9
print(min(lista3)) #0

lista_comprension=[num**3 for num in range(1,11)]
print(lista_comprension) #[1, 8, 27, 64, 125, 216, 343, 512, 729, 1000]


-----esto es un test para visual code - pr