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
