# Proyecto de graduación
## _Aumento de datos utilizando redes generativas para mejorar el reconocimiento y segmentación de imágenes de microscopía óptica._

Este proyecto utiliza el modelo editGAN de Nvidia para generar parejas de imágenes realistas de nematodos y sus respectivas máscaras semánticas, con el propósito de entrenar a un segmentador que sea capaz de reconocer a los pequeños animales en imágenes nuevas. Los cambios realizados en este Fork fueron hechos principalmente para adaptar la arquitectura al conjunto de datos que se tiene a discposición, en términos de su resolución y contenido semántico. Adicionalmente, se desarrollaron nuevos scripts para agilizar el procesamiento de datos entre las diferentes partes del modelo y para producir el conjunto de datos deseado. A continuación se presenta un algoritmo para reproducir el entrenamiento realizado en el proyecto:

1. Entrenar un modelo de StyleGAN2 (la versión vanilla) con el conjunto de nematodos disponible.
2. Convertir el checkpoint de Stylegan2 a formato pytorch (ver ejemplo en https://github.com/dvschultz/ai/blob/master/Convert_pkl_to_pt.ipynb)
3. Poner la dirección del checkpoint resultante en experiments/datasetgan_nema.json, experiments/tool_nema.json y experiments/encoder_nema.json
4. Entrenar el encoder con el comando 

    ```sh
    train_encoder.py --exp experiments/encoder_nema.json
    ```

5. Preparar un pequeño conjunto etiquetado (de aprox. 16 ejemplos, de otra manera se requiere de mucha ram) donde las imágenes estén nombradas según el formato image_#.png y las máscaras según el formato image_mask_#.npy. Para hacer estos cambios automáticamente, ver el script data_processor.py
6. Incluir la dirección del conjunto etiquetado (imágenes y máscaras deben estar en la misma carpeta) en experiments/datasetgan_nema.json, en la variable llamada annotation_mask_path
7. Introducir el conjunto etiquetado en el espacio latente de StyleGAN2 con el comando 
    ```sh
    python train_encoder.py --exp experiments/encoder_car.json --resume *encoder checkppoint* --testing_path data/annotation_car_32_clean --latent_sv_folder model_encoder/nema/training_embedding --test True
    ```
8. Agregar la carpeta resultante a la variable optimized_latent_path en experiments/datasetgan_nema.json
9. Entrenar a DatasetGAN (que es responsable de la segmentación) con el comando
    ```sh
    python train_interpreter.py --exp experiments/datasetgan_nema.json
    ```
    NOTA: Este paso requiere de mucha RAM. De no tener suficiente, una alternativa no ideal es usar memoria virtual para completar el entrenamiento, lo que lo hará muy lento. Para hacer esto es necesario contar con un espacio swap suficientemente grande y utilizar de comando
    
    ```sh
    systemd-run --scope -p MemoryMax=26G python train_interpreter.py --exp experiments/datasetgan_nema.json
    ```
    Esto hace que el programa solo pueda usar MemoryMax de la ram, obligando al sistema a utilizar el espacio swap para el resto que necesite el entrenamiento.
        
10. Utilizar el script ../dataset_gen.py para generar parejas imagen-máscara o para generar una segmentación de imágenes reales de entrada.

11. Entrenar el support vector machine con svm_trainer.py

12. Aplicar el filtro con image_filter.py


Se ha incluido un notebook de jupyter llamado full_system.ipynb donde se realiza todo el proceso paso a paso.

## Utils adicionales
Dos scripts adicionales fueron utilizados durante el proyecto:

- mask_gen.py:
    Utilizado para generar máscaras a partir de un file de anotaciones .json en el formato VIA (ver https://github.com/ox-vgg/via)
- simple_aug.py:
	Genera 7 imágenes adicionales por cada una de entrada con rotaciones de 90&deg; y reflexiones.
- data_processor.py: 
	Una versión consolidada de image_renamer y mask_converter que elige una cantidad determinada de ejemplos aleatorios con sus máscaras para datasetGAN.
- test_image_picker.py:
	Un pequeño programa para elegir al azar una cantidad determinada de ejemplos para usar en las pruebas al final del experimento.
- mosaic_maker.py:
	Un script para generar mosaicos de los resultados.

