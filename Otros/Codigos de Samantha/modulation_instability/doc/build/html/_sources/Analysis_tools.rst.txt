Analysis tools
==============

In this section we briefly describe the available tools for the analysis of the
free surface deformation fields obtained in the framework of this project.


Descomposicion entre moduladora y modulada
------------------------------------------
  A partir de un volumen 3D de datos tipo :math:`h(x,y,t)`, obtener dos
  conjuntos de datos `reales` 3D de la forma :math:`M(x,y,t)` (moduladora o envolvente) 
  y :math:`H(x,y,t)` (alta frecuencia). 

Calculo de la velocidad vertical
--------------------------------
  A partir de un volumen 3D de datos tipo :math:`h(x,y,t)`, obtener otro
  conjunto de iguales caracteristicas pero correspondiente a la velocidad
  vertical en cada punto; i.e., :math:`u_\perp(t) = \partial_t h(x,y,t)`. Para esto se toma
  la serie temporal a punto fijo dada por :math:`h(x_0, y_0, t)`, se la
  ajusta empleando un spline y luego se genera la señal derivada
  :math:`u_\perp(t)` a partir de dichos coeficientes, evitando la amplificacion
  del ruido usual en la derivacion numerica.

Espectros 2D en tiempo
----------------------
  A partir de un volumen 3D de datos tipo :math:`h(x,y,t)`, obtener los espectros
  2D `complejos` en funcion del tiempo, de la forma :math:`E_h(k_x, k_y, t)`.

Espectros 3D
------------
  A partir de un volumen 3D de datos tipo :math:`h(x,y,t)`, obtener los
  espectros `complejos` 3D de la forma :math:`E_h(k_x, k_y, \omega)`. 

Notar que los espectros (tanto 2D como 3D) mencionados anteriormente pueden ser
calculados tanto sobre los conjuntos originales :math:`h(x,y,t)` como sobre :math:`M(x,y,t)` o incluso sobre :math:`H(x,y,t)`.

