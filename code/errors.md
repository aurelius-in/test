Epoch 1/1000
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[7], line 4
      1 #
      3 vae.compile(optimizer='adam')
----> 4 vae.fit(X_synthetic, epochs=1000, batch_size=64) 
      6 #

File /anaconda/envs/azureml_py38_PT_TF/lib/python3.8/site-packages/keras/utils/traceback_utils.py:70, in filter_traceback.<locals>.error_handler(*args, **kwargs)
     67     filtered_tb = _process_traceback_frames(e.__traceback__)
     68     # To get the full stack trace, call:
     69     # `tf.debugging.disable_traceback_filtering()`
---> 70     raise e.with_traceback(filtered_tb) from None
     71 finally:
     72     del filtered_tb

File /tmp/__autograph_generated_file5rkyfum8.py:15, in outer_factory.<locals>.inner_factory.<locals>.tf__train_function(iterator)
     13 try:
     14     do_return = True
---> 15     retval_ = ag__.converted_call(ag__.ld(step_function), (ag__.ld(self), ag__.ld(iterator)), None, fscope)
     16 except:
     17     do_return = False

ValueError: in user code:

    File "/anaconda/envs/azureml_py38_PT_TF/lib/python3.8/site-packages/keras/engine/training.py", line 1160, in train_function  *
        return step_function(self, iterator)
    File "/anaconda/envs/azureml_py38_PT_TF/lib/python3.8/site-packages/keras/engine/training.py", line 1146, in step_function  **
        outputs = model.distribute_strategy.run(run_step, args=(data,))
    File "/anaconda/envs/azureml_py38_PT_TF/lib/python3.8/site-packages/keras/engine/training.py", line 1135, in run_step  **
        outputs = model.train_step(data)
    File "/anaconda/envs/azureml_py38_PT_TF/lib/python3.8/site-packages/keras/engine/training.py", line 993, in train_step
        y_pred = self(x, training=True)
    File "/anaconda/envs/azureml_py38_PT_TF/lib/python3.8/site-packages/keras/utils/traceback_utils.py", line 70, in error_handler
        raise e.with_traceback(filtered_tb) from None
    File "/anaconda/envs/azureml_py38_PT_TF/lib/python3.8/site-packages/keras/engine/input_spec.py", line 295, in assert_input_compatibility
        raise ValueError(

    ValueError: Input 0 of layer "vae_mlp" is incompatible with the layer: expected shape=(None, 96), found shape=(None, 97)
