# Lineamiento
- Estudiar los efectos de la modulación de fase y cuantificarlos.
- Con Samantha vieron que había difusión de la fase, se podría cuantificar y modelar.
- Oscilones.

# Boceto de slides
1. Título.
2. Ondas de Faraday, probablemente contar el 2d. Contar que viene de un forzado paramétrico. Mostrar inestabilidad de Mathieu. Mathieu es un buen modelo a primer orden del sistema.
3. Nosotros queremos medir 1d con cond periódicas de contorno e introducir una deformación de fase. Estudiamos esto como sistema de juguete. Poner refe bibliográfica.
4. Setup experimental. Medición óptica y acelerómetro. Comentar el equipamiento.
5. Resultados del acelerómetro. Los 3 ejes.
6. Perspectivas. Si podemos volver al labo nos ponemos a medir, sino, podemos calibrar el acelerómetro y trabajar con datos no explotados medidos por Samantha.

# Objetivos
- Nos interesa estudiar cómo y en qué regímenes la modulación de fase afecta a los mecanismos de formación de patrones y si la misma es responsable por la emergencia de flujos medios y/o estructuras coherentes.

# Slides
## 1 - Título
- Presentación de nosotros, Pablo y el lugar del trabajo.
- Nombras que es un sistema de juguete para estudiarlo como paradigma de sistemas que muestran un patrón de inestabilidad.

## 2,3,4 - Ondas de Faraday en 2D
- Empezamos explicando que cuando se comienza a excitar verticalmente un fluido se ve el patrón de la primera imagen. Que es el patrón clásico de ondas estacionarias.
- Un tiempo después se puede ver cómo en la superficie comienza a aparece una cierta rugosidad, modulada radialmente.
- Al final se llega a un estacionario donde se forma un patrón con una estructura coherente. Se ven como pequeños "blobs". Estructuras localizadas con una longitud característica a las que se denomina oscilones. En este trabajo Shats y compañía que esta estructura no está formada por una superposición de ondas estacionarias, sino que son ensambles de "quasipartículas". Este resultado no nos interesa estudiarlo en particular pero denota la riqueza del sistema y nos da una buena intuición de cuáles son los modelos teóricos con los que lo podríamos atacar.

## 5 - Agua con proteína vs agua
- Explicamos que en realidad el sistema con agua no tiene una estructura tan rígida sino que los "blobs" se mueven al azar, chocan y se fusionan.

## 6 - Estabilidad de Mathieu
- La ecuación de Mathieu describe una gran variedad de sitemas físicos y en particular sistemas que presentan resonancia paramétrica como la que presentamos recién. Esta es una ecuación homogénea de segundo orden que debe resolverse computacionalmente. La resolución numérica de la ecuación diferencial es muy interesante pero para los fines de esta charla lo que vale la pena remarcar es el resultado que se muestra en el gráfico de la izquierda. En este gráfico vemos las regiones de estabilidad de la solución donde el eje vertical depende de los parámetros del fluido y el eje horizontal de la frecuencia de excitación.
- Tener en cuenta que este modelo no incluye la viscosidad del sistema, ya que estamos trabajando con agua. Pero de todos modos existe un desarrollo teórico al respecto que muestra resultados similares a éste pero con las puntas redondeadas.
- Este gráfico nos muestra que para cualquier fluido existen siempre frecuencias donde este genera ondas de Faraday.

## 7, 8 - Montaje experimental
- Fuimos dos veces al labo. En la primera nos familiarizamos con el equipo, que por suerte ya está en gran parte desarrollado y nos permite realizar mediciones.
- En la segunda vez ya pusimos en funcionamiento el sistema y nos centramos sobre todo en el acelerómetro del montaje. Después mostramos algunos resultados de esto.
- Explicamos por qué cosas está formado.

## 9, 10 - Medición de la superficie libre
- El objetivo es medir la superficie libre del fluido; esto es, medir la altura en cada punto.
- Para hacer esto usamos lo que llama perfilometría por transformada de Fourier (FTP).
	- Proyectamos sobre el fluido estacionario un patrón con una frecuencia espacial conocida y restamos el fondo.
	- Al deformar la superficie libre este patrón también lo hace y transformando Fourier y comparando con el estacionario podemos obtener la fase relativa. Esta fase relativa tiene toda la información de la altura de la superficie, y si la distancia entre el proyector y la cámara es mucho menor a la distancia de la cámara a la superficie libre, la fase es proporcional a la altura.

## 11 - Acelerómetro
- En lo que estuvimos trabajando las últimas dos semanas fue en programar el Arduino para optimizar la frecuencia de adquisición del acelerómetro.
- A la derecha se puede ver el acelerómetro con las conexiones que se hicieron y a la izquierda las mediciones para el sistema oscilado a 20Hz.
- Vemos que las variaciones en $x$ e $y$ son despreciables respecto a las de $z$ y que las mediciones del eje $z$ oscilan con la frecuencia de excitación del sistema como es esperado.
- Como vemos a las mediciones le falta un factor de escala proveniente de la calibración. Este factor es solo un número que sirve para redimensionar la escala, por lo que se puede calcular a posteriori.

## 12 - Perspectivas
- Recientemente terminamos de realizar un graficador en vivo, lo cual nos permite monitorizar activamente el experimento a la hora de hacer mediciones y facilita la calibración del acelerómetro.
- Hoy vamos a ir al labo a tomar las primeras mediciones de la deformación de la superficie libre. Con estas mediciones ya podemos poner en práctica lo que estuvimos aprendiendo las primeras semanas sobre extracción de la onda moduladora mediante transformada de Hilbert.
- Un pequeño inconveniente a trabajar es buscar una forma eficiente de realizar la transformada de Fourier a los datos del acelerómetro ya que los puntos que obtenemos de este no estás equiespaciados y la FFT rip.
- Un siguiente y breve paso va a ser trabajar sobre la adquisición y guardado de los datos en bases de datos eficientes. Tenemos en mente el protocolo HDF5.
- Luego de tener todos los datos medidos y guardados eficientemente el siguiente propósito es proponer un modelo de Schrödinger no lineal como solución y trabajar en algún algoritmo razonable para estimar los parámetros. Tenemos en mente estimación bayesiana o SINDY (que es un método para estimar la forma funcional de las ecuaciones diferenciales que dictan la dinámica de una serie de datos).
- Como comentario adicional mencionar que hay muchos datos tomados y no explotados por Kucher.


## Dudas

- Unidades del output del acelerómetro.

