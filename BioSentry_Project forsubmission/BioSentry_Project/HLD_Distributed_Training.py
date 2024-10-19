strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = models.Sequential([
        # CNN layers
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Now training is distributed over multiple GPUs
model.fit(train_generator, epochs=10)
