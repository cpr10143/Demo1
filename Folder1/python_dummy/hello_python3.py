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
print(frase3*2) #repertir

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
print(len("Hola mundo"))

frase='Esto es para saber que posicion tiene la palabra dentro d euna frase - saber, si saber1'
print(frase.count ('la',3,100)) #2
print(frase.find ('sa')) #posicion
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
#>> ['Hola mundo!', 'Hello world!']
"Hola mundo. Hello world!".partition(" ") #retorna una tupla de tres elementos: el bloque de caracteres anterior a la primera ocurrencia del separador, el separador mismo, y el bloque posterior
#>> ('Hola', ' ', 'mundo. Hello world!')
"Hola mundo. Hello world!".rpartition(" ") #opera del mismo modo que el anterior, pero partiendo dederecha a izquierda
#>> ('Hola mundo. Hello', ' ', 'world!')
", ".join(["C", "C++", "Python", "Java"]) #debe ser llamado desde una cadena que actúa como separador para unir dentro de una misma cadena resultante los elementos de unalista
#>> 'C, C++, Python, Java'

print('-------------------**********------------------------')
print('-------------------****EJER CURSO VIDEO STRING******------------------------')
print('-------------------**********------------------------')

lista=['esto' ,'es', 'para', 'hacer', 'un', 'ejercicio']
#ejer1=lista.join(',') #error 
ejer1=' ,, '.join(lista) # OK 
print(ejer1)
cadena="Si la implementación es dificil de explicar, puede que sea una mala idea."
ejer2=cadena.replace('dificil', 'facil').replace('mala','buena')
print(ejer2)
cadena2='''Tierra mojada,
mis recuerdos de viaje
entre las lluvias'''
print(cadena2)
ejer3=cadena2.__contains__('agua')
print("agua " +str(ejer3) +" existe en haiku") ##false
#MEJOR OPCION !!!
print('agua' in cadena2)  ##false
print('agua' not in cadena2)  ##True


#termino video 64 y tengo que empezasr por el ejer 69
#--69
ejer69='Repetición '
print(ejer69*15)

nombre="Carina"
#nombre[0]='K' --> esto da error porque los str son inmudatebles
nombre2='Karina'
print(nombre+nombre2)
print(nombre*5) #CarinaCarinaCarinaCarinaCarina

ejer71='electroencefalografista'
print(len(ejer71)) #23

ejer73=[1,'ooo',2,True, 5.5]
print(ejer73)

ejer75=['manzana','banana','mango', 'cereza' , 'sandía']
popped=ejer75.pop()
print(ejer75) #['manzana', 'banana', 'mango', 'cereza']
print(popped) #sandía
popped2=ejer75.pop(2)
print(ejer75) #['manzana', 'banana', 'cereza']
print(popped2) #mango

ejer74= ["avión", "auto", "barco", "bicicleta"]
ejer74.append('mottoooo')
print(type(ejer74)) #<class 'list'>
print(ejer74) #['avión', 'auto', 'barco', 'bicicleta', 'mottoooo']
ejer74.insert(-2,'motillo')
print(ejer74) #['avión', 'auto', 'barco', 'motillo', 'bicicleta', 'mottoooo']
ejer74.append(101)
print(type(ejer74)) #<class 'list'>
print(len(ejer74)) #7
print(ejer74[0:2]) #['avión', 'auto']
print(ejer75+ejer74)
print(ejer75.sort()) # sort() en Python ordena la lista en el sitio y no devuelve la lista ordenada. Devuelve None.
print(type(ejer75.sort())) #<class 'NoneType'>s
print(ejer75) #['banana', 'cereza', 'manzana']
ejer75.reverse()
print(ejer75) #['manzana', 'cereza', 'banana']


ejerLibro=["avión", "auto", "barco", "bicicleta"]
ejerLibro.remove('auto') ##Si no sabemos la posición en la lista del elemento a borrar
#Con REMOVE eliminamos por np¡ombreeeeee, utillll!!!!!!!!
print(ejerLibro) # =["avión",  "barco", "bicicleta"]

#**Vease hoja ejer libro 3


print('-------------------------------**********------------------------------------')
print('-------------------****EJER CURSO VIDEO DICCIONARIOS {} ******------------------------')
print('-------------------------------**********------------------------------------')

diccionario={'clave1':'valor1','clave2':'valor2'} 
#las claves deben ser unicos, los valroes peuden repetirse. Las claves no
print(type(diccionario))
print(diccionario)
resul=diccionario['clave1']   
print(resul)        

dic={'c1':55,'c2':[10,20,30],'c3':{'s1':100,'s2':290}}
print(dic['c2']) #[10, 20, 30]
print(dic['c2'][1]) #20
print(dic['c3']) #{'s1': 100, 's2': 290}
print(dic['c3']['s2']) #290

dic2={'c1':['a','b','c'],'c2':['d','e','f']}
print(dic2['c2'][0].upper()) #D

dic3={1:'a',2:'b'}
print(dic3)
dic3[3]='c' #añadir al diccionario 
print(dic3) #{1: 'a', 2: 'b', 3: 'c'}
#sobreescribir
dic3[2]='B'
print(dic3) #{1: 'a', 2: 'B', 3: 'c'}
print(dic3.keys()) #dict_keys([1, 2, 3])
print(dic3.values()) #dict_values(['a', 'B', 'c'])
print(dic3.items()) #dict_items([(1, 'a'), (2, 'B'), (3, 'c')])

ejer77={'nombre':'Karen','apellido':'Jurgens','edad':35,'ocupacion':'Periodista'}
print(ejer77.items()) #dict_items([('nombre', 'Karen'), ('apellido', 'Jurgens'), ('edad', 35), ('ocupacion', 'Periodista')]) 

ejer78 = {"valores_1":{"v1":3,"v2":6},"puntos":{"points1":9,"points2":[10,300,15]}}
print(ejer78['puntos']['points2'][1] ) #300

ejer79=ejer78
ejer79['pais']='Colombia'
ejer79['edad']=36
ejer79['ocupacion']='Editora'
print(ejer79) #{'valores_1': {'v1': 3, 'v2': 6}, 'puntos': {'points1': 9, 'points2': [10, 300, 15]}, 'pais': 'Colombia', 'edad': 36, 'ocupacion': 'Editora'}

print('-------------------------------**********------------------------------------')
print('-----****EJER CURSO VIDEO tuplas +eficientes,inmutables,-espacio en memoria ******----')
print('-------------------------------**********------------------------------------')

tupla=(1,2,3,4)
tupla2=1,2,3,4 #se puede construir sin parentesis
print(type(tupla)) #<class 'tuple'>
print(tupla[0]) #1
print(tupla[-1]) #4
#tupla[0]=5 # ERROR , tuple no soporta asignacion de valores como los str
#anidar
tupla3=(1,2,(10,20),4) 
print(tupla3[2][0]) #10
var=list(tupla3)
print(type(var)) # <class 'list'>
print(var) #[1, 2, (10, 20), 4]

tu=(1,2,3)
x,y,z=tu #para asignar los valores de la tupla, tienen que tener la misma 
         #cantidad de elementos
# y,z=tu #esto da error    !!   
print(x,y,z) #1 2 3
print(len(tu)) #3
print(tu.count(2)) #1 aparece 1 vez
print(tu.index(1)) # 0 posicion

ejer81 = (1, 2, 3, 2, 3, 1, 3, 2, 3, 3, 3, 1, 3, 2, 2, 1, 3, 2)
print(ejer81.count(2)) #6
ejer83=(1, 2, 3, 4)
a, b, c, d=ejer83
print(a, b, c, d) #1 2 3 4
ejer82 = (1, 2, 3, 2, 3, 1, 3, 2)
mi_lista=list(ejer82) #de tupla a lista
print(mi_lista) #[1, 2, 3, 2, 3, 1, 3, 2]


print('-------------------------------**********------------------------------------')
print('---------------------------****EJER CURSO VIDEO SETs ******----')
print('-------------------------------**********------------------------------------')

#set puede ser set([1,2,3,4,5]) tinen q ir como 1 elemento o {1,2,3,4,5}
#en un set no puede haber elemnetos repetidos. Son uno. No se ordena en indices
#no se puede reorganizar , inmutable
#set1=set(1,2,3,4,5) ERROR DE TIPO,  se puede arreglar ((1,2,3,4,))
set1=set([1,2,3,4,5])
print(type(set1)) #<class 'set'>
print(set1) #{1, 2, 3, 4, 5} 
set2=set((1,2,3,4,5)) ###ERORROR RRR con {}
print(type(set2)) #<class 'set'>
print(set2) #{1, 2, 3, 4, 5} 
set3={1,2,3,4,5}
print(type(set3)) #<class 'set'>
print(set3) #{1, 2, 3, 4, 5} 
#print(set1[0]) #en los set no  se puede hacer esto ERROR

set4={1,2,3,4,5,1,1,2,2,3,3,4,6} #NO PUEDE HABER REPETICIONES
print(set4) #{1, 2, 3, 4, 5, 6} 
#NO adminte listas dentro [], ni diccionarios, pero SI tuplas porque son inmutables 
#por eso 
set5=set((1,2,3,4,(1,2,3) ,1,1,1))
print(set5) #{1, 2, 3, 4, (1, 2, 3)}
set6=set((1,2,3,4,5)) #OJO Que no se olvide la palabra SET
print(type(set6)) #<class 'set'>
print(len(set6)) #5
print(2 in set6) #true
#union de sets
set7={5,6,7}
set8=set6.union(set7)
print(set8) #{1, 2, 3, 4, 5, 6, 7}
print(set7|set8) #se puede concanetar tb con | {1, 2, 3, 4, 5, 6, 7}
#opes con set
# mi_set_a.add(5) mi_set_a.clear() mi_set_c = mi_set_a.copy()
# mi_set_c = mi_set_a.difference(mi_set_b) difference
#(set) retorna el set formado por todos los elementos queúnicamente existen en el set A 
# mi_set_a.difference_update(mi_set_b)
#mi_set_a.discard("tres")  remueve un elemento del set
#mi_set_c = mi_set_a.intersection(mi_set_b)retorna el set formado por todos los elementos queexisten en A y B simultáneamente.
# ETC mil funcones
#
#Elimina un elemento al azar del siguiente set, utilizando métodos de sets.

sorteo = {"Camila", "Margarita", "Axel", "Jorge", "Miguel", "Mónica"}
ganador=sorteo.pop()
print(sorteo) #{'Axel', 'Margarita', 'Camila', 'Jorge', 'Mónica'}
print(ganador) #Miguel
sorteo.add("Damián")
print(sorteo) #{'Damián', 'Margarita', 'Miguel', 'Jorge', 'Axel', 'Camila'}



print('-------------------------------**********------------------------------------')
print('---------------------------****EJER CURSO VIDEO BOOLEAN ******----')
print('-------------------------------**********------------------------------------')

var1=True
var2=False
print(type(var1)) #<class 'bool'>
print(type(var2))#<class 'bool'>
num=5 > 2+3
print(type(num))#<class 'bool'>
print(num) #False
num2=bool(5>6)
print(num2) #False (es lo mismo que poenr la expresión)
num2=bool() #esto da falso, por si quiero inicializar una var a false

lista1=[1,2,3,4,5]
control= 5 in lista1
print(control) #True

ejer89=4 == 8 
print(ejer89)
ejer90=(17834/34) > 87*56
print(ejer90) #false or print((17834/34)>87*56)
ejer91=25**0.5 == 5 #raiz cuadrada
print(ejer91) #true







 

