** First model
Layers: <br>
  * dense layer 128
  * dense layer 128
  * dense layer 10
accuracy: 96.67999744415283% <br>
accuracy_rounded: 96.68% <br>
custom_test_score: 4/10 <br>

** Second model
Layers: <br>
  * dense layer 256
  * dense layer 128
  * dense layer 64
  * dense layer 10
accuracy: 97.18999862670898 <br>
accuracy_rounded: 97.19% <br>
custom_test_score: 4/10

** Third model
Convolutional Neural Networks
Layers: <br>
  * Conv2D 32
  * MaxPooling2D
  * Conv2D 64
  * MaxPooling2D
accuracy: 98.76000285148621% <br>
accuracy_rounded: 98.76% <br>
custom_test_score: 8/10

** Fourth model
Convolutional Neural Networks trained with better data set
Layers: <br>
  * Conv2D 32
  * MaxPooling2D
  * Conv2D 64
  * MaxPooling2D
  * Dense 128
  * Dropout 0.5
  * Dense 10
accuracy: 99.01999831199646% <br>
accuracy_rounded: 99.02% <br>
custom_test_score: 8/10
