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

print('\n')
print('-------------------****EJER LIBRO******------------------------')
print('---------------------------------------------------------------')

print('\n')
coches=['mercedes','toyota','seat','mazda']
coches.insert(1,'fiat')
print(coches)


ejerLibro=["avión", "auto", "barco", "bicicleta"]
ejerLibro.remove('auto') ##Si no sabemos la posición en la lista del elemento a borrar
#Con REMOVE eliminamos por np¡ombreeeeee, utillll!!!!!!!!
print(ejerLibro) # ['avión', 'barco', 'bicicleta']

ejer3_4=["ana", "luis", "bea"]
print(f"{ejer3_4[0].title()} estás invitad@ a la fiesta! :) ")
print(f"{ejer3_4[1].title()} estás invitad@ a la fiesta! :) ")
print(f"{ejer3_4[2].title()} estás invitad@ a la fiesta! :) ")

ejer3_5_ko=ejer3_4.pop(1)
print(f"Finalmente, {ejer3_5_ko.title()} no podrá asistir a la fiesta! :( ")
ejer3_5_new='paca'
ejer3_4.append(ejer3_5_new)
print(ejer3_4)

#ejer3_6
ejer3_4.insert(0,'jaimito')
ejer3_4.insert(2,'lucas')
ejer3_4.append('ruben')
print(ejer3_4) #['jaimito', 'ana', 'lucas', 'bea', 'paca', 'ruben']
print('finalmente irán a la cena ',len(ejer3_4), ' personas') #6
ejer3_7= ejer3_4
print(type(ejer3_7))
del ejer3_7[1]
print(ejer3_7) #['jaimito', 'lucas', 'bea', 'paca', 'ruben']
del ejer3_7[:]
print(ejer3_7) #[]

cars=['bmw','audi','subaru','toyota','seat' ]
cars.sort()
print(cars) #['audi', 'bmw', 'seat', 'subaru', 'toyota']
cars.sort(reverse=True)
print(cars) #['toyota', 'subaru', 'seat', 'bmw', 'audi']

cars2=['bmw','audi','subaru','toyota','seat' ]
print('el orden original es :')
print(cars2) #['bmw', 'audi', 'subaru', 'toyota', 'seat']
print('en orden para mostrar pero no interno sería ')
print(sorted(cars2)) #['audi', 'bmw', 'seat', 'subaru', 'toyota']
print('porque si volvemos a imprimir, vemos el orden original')
print(cars2) #['bmw', 'audi', 'subaru', 'toyota', 'seat']

cars2.reverse() #invierte el orden de las posicones, no reordena
print(cars2) #['seat', 'toyota', 'subaru', 'audi', 'bmw']
len(cars2)
