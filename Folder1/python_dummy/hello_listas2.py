mi_lista=['a','b','c']
print(type(mi_lista))

lista2=['hola',55,6.1]
print(type(lista2))
print(len(lista2))
print('-------------------**********------------------------')
print(lista2[0])
print(lista2[0:1])
print(lista2[0:2])
print('-------------------**********------------------------')
lista3=['d','e','f']
print(mi_lista+lista3)

lista4=mi_lista+lista3
print(lista4)
print('-------------------**********------------------------')
lista4[0]='alfa'
print(lista4)

lista4.append('g') #solo añade al final de la lista
print(lista4)

lista4.pop() #elimina el uñtimo elemento de la lista o se lo especificas 

eliminado=lista4.pop(0) 
print(lista4)
print(eliminado)
print('-------------------**********------------------------')
lista5=['d','e','f', 'a','c','b']
print(lista5)
lista6=lista5.sort() #Funcion que no devuelve nada ,esto da un nontype
print(type(lista6)) #esto da un nontype que es que no tiene valor , no confirndir con que sea 0
print(lista5)
print('-------------------**********------------------------')
lista5.reverse() #ordena al reves
print(lista5)

print(split('hola como estas?' , '/'))

print('-------------------****EJER LIBRO******------------------------')
coches=['mercedes','toyota','seat','mazda']
coches.insert(1,'fiat')
print(coches)