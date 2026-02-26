name='Ana '
print(name)
 
print(name.__len__()) #suma el espacio
name.rstrip() #quito espacios pero no almaceno
print(name.__len__()) #sigue siendo 4
name=name.rstrip() #actualizo con ello limpio
print(name.__len__()) #ya es 3

#lstrip() left ,rstrip() right , strip() both

nostarch_url='https://nostrach.com'
nostarch_url.removeprefix('https://')
simple_url =nostarch_url.removeprefix('https://')
print(simple_url)

sms="One of the python's strength is its deverse community"
print(sms)

print("Hola "+ input("Como te llamas? ").strip() + ", Te gustaría aprender python hoy?")

nombre=input("Segundo nombre,pls?")
print (nombre.lower() + " " + nombre.upper() + " "+nombre.title() + " " + nombre.strip())

print("Mi abuela un día dijo, 'A buen entendedor, pocas palabras bastan")



