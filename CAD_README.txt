----------------------------------
Collective Activity Dataset Ver.2
----------------------------------

Contents : 2 additional collective activities Dancing and Jogging.

Annotation

Every 10th frame in all video sequences was annotated with image location of person, activity id, and pose direction.
Frame number, X, Y, WIDTH, HEIGHT, CLASS ID, POSE ID.
ex. 001       366     168     106     212     5       3
    001     512     190     98      195     5       3
    001     440     187     84      167     5       3
    001     339     191     83      165     5       3

CLASS ID 
1. NA, 2. Crossing, 3. Waiting, 4.Queuing, 5. Walking (not use), 6. Talking, 7. Dancing, 8. Jogging

POSE ID
1. Right 2. Front-right 3. Front 4. Front-left 5. Left 6. Back-left 7. Back 8. Back-right

Note : 
seq45~51, seq52~53, seq54~58 must be treated as "one video" to prevent contamination in learning due to the similarity.
