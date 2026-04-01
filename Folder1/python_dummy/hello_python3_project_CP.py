#Proyecto del Día 3
# La consigna es la siguiente: vas a crear un programa que primero le pida al usuario que
# ingrese un texto. Puede ser un texto cualquiera: un artículo entero, un párrafo, una frase, un
# poema, lo que quiera. Luego, el programa le va a pedir al usuario que también ingrese tres
# letras a su elección y a partir de ese momento nuestro código va a procesar esa información
# para hacer cinco tipos de análisis y devolverle al usuario la siguiente información:

texto=input('Ingrese un texto cualquiera que le guste: ')
texto.lower()
#lista=list( input('Ahora ingrese 3 letras separadas por comas (ej: a,w,t) : ').split(','))
lista= input('Ahora ingrese 3 letras separadas por comas (ej: a,w,t) : ').split(',')
print(type(lista))
print(lista)
print(len(texto))
#****************************************************************
# 1 ¿cuántas veces aparece cada una de las letras que eligió?
#****************************************************************
numVeces= texto.count(lista[0].lower())
print(numVeces )
#print(frase.count ('la',3,100)) #2
#print(frase.find ('sa')) #posicion
print(lista[0] , ' aparece ' , texto.count(lista[0].lower()) , ' veces en el texto inicial')
print(lista[1] , ' aparece ' , texto.count(lista[1].lower()) , ' veces en el texto inicial')
print(lista[2] , ' aparece ' , texto.count(lista[2].lower()) , ' veces en el texto inicial')

#****************************************************************
# 2 ¿decir al usuario cuántas palabras hay a lo largo de todo el texto?
#****************************************************************
lista=texto.split()
print('La cantidad total de palabras que tiene el texto inicial es de: ', len(lista))
 
#****************************************************************
# 3 ¿cuál es la primera letra del texto y cuál es la última?
#****************************************************************
print('La primera letra del texto es: ' ,lista[0][0])
print('La última letra del texto es: ' ,lista[-1][-1])

#****************************************************************
# 4 ¿cómo quedaría el texto si invirtiéramos el orden de las palabras?
#****************************************************************
lista.reverse()
print('El texto al revés quedaría así: \n' , ' '.join(lista))

#****************************************************************
# 5 el sistema nos va a decir si la palabra “Python” se encuentra dentro del texto
#****************************************************************
pal='Pyhton'
print('Existe la palabra ' ,pal, ' en el texto inicial? ' , pal in texto)

#MEJORAS hACER UN DICCIONARIO EN EL PUNTO 5 Y REEMPLAZAR EL TRUE FALSE
buscarPal= pal in texto
dic={True:'si', False:'no'}
print((f"La palabra {pal} , {dic[buscarPal]} se encuentra en el texto"))
