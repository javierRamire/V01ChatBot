import h5py

# Abre el archivo .h5 en modo de lectura
with h5py.File('chatbot_model.h5', 'r') as archivo:
    # Itera sobre los elementos del archivo
    print("Datasets dentro del archivo:")
    def listar_datasets(nombre, objeto):
        if isinstance(objeto, h5py.Dataset):
            print(nombre)
    archivo.visititems(listar_datasets)



# Abre el archivo .h5 en modo de lectura
with h5py.File('chatbot_model.h5', 'r') as archivo:
    # Puedes listar los grupos y datasets dentro del archivo
    print("Grupos y datasets dentro del archivo:")
    for nombre in archivo:
        print(nombre)

    # Accede a un dataset específico
    dataset = archivo['model_weights/dense/bias']

    # Lee los datos del dataset
    datos = dataset[:]
    print("Datos del dataset:")
    print(datos)
