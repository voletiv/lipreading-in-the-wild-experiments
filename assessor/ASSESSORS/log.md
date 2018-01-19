- Changed SyncNet input images to have values between 0 and 255

29

30

31

32

33

- Trying Residual connection, with dropout

34

- Removed dropout in residual connection

35

- Removed residual connection, reduced input fcs and LSTMs to 4

36

- Removed min_lr from ReduceLROnPlateau, using RMSprop

37

- Using adam with decay and ReduceLROnPlateau (with no min_lr)

38

- Decay was too much - No lr decay in adam

39

- Overfitting. ReduceLROnPlateau with patience=3 (was 5)

40

- Slight overfitting
- Reduced to 2, 2, 2

41

- Very bad! No training!
- Back to good configuration (8, 8, 8, 4), without residual

42

- With trainable residual

43

- Added 50 samples per word from LRW_train

44

- "equal_classes" deactivated

45

- Found LRWtrain, found correct syncnet_preds
- Use LRWtrain of 200*500 samples, contrastive loss! Euclidean distance b/w Syncnet->LSTM->64 bit and lipreader_preds->->64-bit

46

- Reduce LR rate from 1e-3 to 1e-5, with LR_decay 1e-7

47

- Not using contrastive loss! fcs 64, 32, LR 1e-4, 0 decay

48

- fcs 16, 8

49

- Equal clases! LR 1e-4, decay 1e-7

50

- LR 5e-5, decay 1e-6

51

- Not using LRWtrain

52

- More fcs - 64, 32

53

- fcs 128, 64, using head_pose

54

- fcs 128, 32, dropouts 0.5

55

- lr 5e-5, decay 5e-7

56

- tanh-neg-relu at end, instead of sigmoid; LR 1e-4, decay 5e-7

57

- decreased dropout to 0.2; increased LR to 1e-3, decay 1e-7

58

- 
