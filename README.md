# ActivityRecognitionin
Activity Recognitionin Base on GRU Networks

---

# DATASET

This milan dataset of the WSU CASAS smart home project contains sensor data that was collected in the home of a volunteer adult.  The residents in the home were a woman and a dog.
The woman's children visited on several occasions.

The following activities are annotated within the dataset. The number in parentheses is the number of times the activity appears in the data.

Chores (23)
Bed-to-Toilet (89)
Chores (23) 
Desk_Activity (54)
Dining_Rm_Activity (22)
Eve_Meds (19) 
Guest_Bathroom (330) 
Kitchen_Activity (554)
Leave_Home (214)  
Master_Bathroom (306)
Meditate (17)
Watch_TV (114)
Sleep (96)
Read (314)   
Morning_Meds (41)   
Master_Bedroom_Activity (117) 

The sensor events are generated from motion sensors (these sensor IDs begin with "M"), door closure sensors (these sensor IDs begin with "D"), and
temperature sensors (these sensor IDs begin with "T").
The layout of the sensors in the home is shown in the file milan.jpg.

---

# Comparison

## Naive Bayes

               precision    recall  f1-score   support

            0   0.789474  0.652174  0.714286        23
            1   0.000000  0.000000  0.000000        17
            2   0.714286  0.833333  0.769231        30
            3   0.333333  0.500000  0.400000         8
            4   0.933333  0.636364  0.756757        22
            5   0.915094  0.898148  0.906542       108
            6   0.000000  0.000000  0.000000         2
            7   0.975610  1.000000  0.987654        40
            8   0.937500  1.000000  0.967742        60
            9   0.887324  0.984375  0.933333        64
           10   0.750000  0.966102  0.844444        59
           11   0.750000  0.900000  0.818182        10
           12   0.000000  0.000000  0.000000         4
           13   0.000000  0.000000  0.000000         2
           14   0.000000  0.000000  0.000000         4

    micro avg   0.847682  0.847682  0.847682       453
    macro avg   0.532397  0.558033  0.539878       453
 weighted avg   0.806688  0.847682  0.822448       453
 
 ## SVM
 
                precision    recall  f1-score   support

            0   0.789474  0.652174  0.714286        23
            1   0.000000  0.000000  0.000000        17
            2   0.622222  0.933333  0.746667        30
            3   0.300000  0.375000  0.333333         8
            4   1.000000  0.636364  0.777778        22
            5   0.870690  0.935185  0.901786       108
            6   0.000000  0.000000  0.000000         2
            7   1.000000  0.975000  0.987342        40
            8   0.967742  1.000000  0.983607        60
            9   0.939394  0.968750  0.953846        64
           10   0.787879  0.881356  0.832000        59
           11   0.750000  0.900000  0.818182        10
           12   0.000000  0.000000  0.000000         4
           13   1.000000  1.000000  1.000000         2
           14   0.500000  0.250000  0.333333         4

    micro avg   0.852097  0.852097  0.852097       453
    macro avg   0.635160  0.633811  0.625477       453
 weighted avg   0.819933  0.852097  0.830372       453

## GRU

              precision    recall  f1-score   support

         0.0   0.772727  0.739130  0.755556        23
         1.0   0.571429  0.705882  0.631579        17
         2.0   0.714286  0.833333  0.769231        30
         3.0   0.375000  0.375000  0.375000         8
         4.0   0.937500  0.681818  0.789474        22
         5.0   0.935780  0.944444  0.940092       108
         6.0   0.000000  0.000000  0.000000         2
         7.0   0.975610  1.000000  0.987654        40
         8.0   0.983607  1.000000  0.991736        60
         9.0   0.904762  0.890625  0.897638        64
        10.0   0.887097  0.932203  0.909091        59
        11.0   0.769231  1.000000  0.869565        10
        12.0   1.000000  0.250000  0.400000         4
        13.0   1.000000  0.500000  0.666667         2
        14.0   0.000000  0.000000  0.000000         4

   micro avg   0.878587  0.878587  0.878587       453
   macro avg   0.721802  0.656829  0.665552       453
weighted avg   0.873248  0.878587  0.871555       453

