﻿# ComplexNet文档 #
 **keras_extension**
 第一行的`self`是每个类方法必备的参数，指代类本身，`units`是本层输出维度（如全连接层代表输出神经元数）。
 第二行分别是本层的核（即权重，keras中称其为核，而把权重和偏置统称为权重）和偏置的初始化接口，在`keras.initializers`中有详细的初始化类及其说明。一般全连接层默认He初始化，卷积默认Xiver初始化。如果需要调用其他的初始化类或者要改写，可以去了解`keras.initializers`底层代码。
 第三行分别是核、偏置和输出值的正则化方式，详细和见`keras.regularizers`。
 类似的第四行分别是核和偏置的限制，如限制非负或者行（列）的模2等于1等等。
 最后一行是一个字典型变量，用于传输一些可省缺变量，常见的有指定输入张量形状的`input_shape=(2, 3)`这种用法。（__注意__：输入这些省缺变量时无需使用字典形式即`{'input_shape': (2, 3)}`这种形式）