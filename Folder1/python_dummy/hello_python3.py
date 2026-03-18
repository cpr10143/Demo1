frase='Esto es para saber que posicion tiene la palabra dentro d euna frase - saber, si saber1'
print('Busco una palabra; ' + str(frase.rindex('saber')))
print(frase.index('saber'))
print(frase.index('saber',15)) #si no existe el valor da ERRORRRRRRRRRRRRR 

print(frase.rfind("saber1")) #si no existe el valor este NO da error, da -1
print(frase.find("sabber")) #si no existe el valor este NO da error, da -1

palabra='Ordenador'
print(palabra[4])
print(palabra[-1]) ## posiciones en reverso es 0 7 6 5 4 3 2 1 

frase="Controlar la complejidad es la esencia de la programación"
print(frase.upper())
print(frase.lower())

print(frase.split('a')) #le digo por donde splitea , en este caso cada a, corta 
print(frase.lower())
frase3='aaaaaaaaggaaaaaaaaaaaaa bbbbbbbbbggbbbbbggbbbbb '
frase4=' || '.join([frase, frase3])
print(frase4)
print(frase3*2)

print('''escribo con lineas 
      distintas''')

result=frase.find('gg')
resultado=frase.index

result=frase3.replace('gg', 'iiii')
print(result)

frase2='ABCDEFGHIJKLMNÑOPQRSTUVWXYZ'
fragmento=frase2[2:7]
print(fragmento )

print (frase2[2])
print (frase2[2:])
print (frase2[:7])
print (frase2[2:15:3])
print (frase2[::3])
print (frase2[::-1]) #orden inverso


#ejer: slicing
texto="Es genial trabajar con ordenadores. No discuten, lo recuerdan todo y no se beben tu cerveza"
print(texto[::-1]) #
print(texto[8::3]) #
print(texto[:8]) #


"C:/python36/python.exe".rfind("/")
"Hola mundo".startswith("Hola")
"abc123".isdigit()
"1234".isnumeric()
"1234".isdecimal()
"abc123".isalnum()
"abc123".isalpha()
"abcdef".islower()
"ABCDEF".isupper()
"Hola \t mundo!".isprintable()
"Hola mundo".isspace()
"Hola mundo".__len__()

frase='Esto es para saber que posicion tiene la palabra dentro d euna frase - saber, si saber1'
print(frase.count ('la',3,100)) #2
print(frase.find ('sa'))
print(frase.startswith ('esto')) # esto >> false  Esto >> True
print(frase.isdigit ()) # False
print("123".isdigit()) # true 
print("abc123".isalnum()) #true /alfanuméricos
print("abc123".isalpha()) #False /alfabéticos
print(frase.islower ()) #false
print(frase.isupper ()) #false
print("ABCDEF".isupper ()) #true
print("Hola \t mundo!".isprintable()) #false
print(frase.isspace ()) #false
print('   '.isspace ()) #true /si todos son espacios
frase='Esto es para saber splitear  y hacer es otras cositas chulas es llo kiiooo'
print(frase.split ('es')) #para separar por lo que queramos en este aso es, y va a una lista

"hola mundo".capitalize()
"Hola mundo".encode("utf-8") #codifica la cadena con el mapa de caracteres especificado yretorna una instancia del tipo bytes
"Hola mundo".replace("mundo", "world")
"Hola Mundo!".lower()
"Hola Mundo!".upper()
"Hola Mundo!".swapcase() #mayúsculas por minúsculas y viceversa
"  Hola mundo!   ".strip() #strip( ), lstrip( ) y rstrip ( )remueven los espacios en blanco
"Hola".center(10, "*") # >> '***Hola***' center( ),ljust( )y rjust( ) alinean una cadena en el centro, la izquierda o la derecha
"Hola mundo!\nHello world!".splitlines() #splitlines( ) divide una cadena con cada aparición de un salto de línea
>> ['Hola mundo!', 'Hello world!']
"Hola mundo. Hello world!".partition(" ") #retorna una tupla de tres elementos: el bloque de caracteres anterior a la primera ocurrencia del separador, el separador mismo, y el bloque posterior
>> ('Hola', ' ', 'mundo. Hello world!')
"Hola mundo. Hello world!".rpartition(" ") #opera del mismo modo que el anterior, pero partiendo dederecha a izquierda
>> ('Hola mundo. Hello', ' ', 'world!')
", ".join(["C", "C++", "Python", "Java"]) #debe ser llamado desde una cadena que actúa como separador para unir dentro de una misma cadena resultante los elementos de unalista
>> 'C, C++, Python, Java'