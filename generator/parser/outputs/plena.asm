
 ===== PLENA ASM DUMP =====
graph id: 139927868471776
placeholder  ; L_fn_modules_ln1_parameters_weight_ ; out=shape=(64,) dtype=torch.float32 device=cpu
placeholder  ; L_fn_modules_ln1_parameters_bias_ ; out=shape=(64,) dtype=torch.float32 device=cpu
placeholder  ; L_args_0_ ; out=shape=(2, 128, 64) dtype=torch.float32 device=cpu
placeholder  ; L_fn_modules_fc1_parameters_weight_ ; out=shape=(256, 64) dtype=torch.float32 device=cpu
placeholder  ; L_fn_modules_fc1_parameters_bias_ ; out=shape=(256,) dtype=torch.float32 device=cpu
placeholder  ; L_fn_modules_fc2_parameters_weight_ ; out=shape=(64, 256) dtype=torch.float32 device=cpu
placeholder  ; L_fn_modules_fc2_parameters_bias_ ; out=shape=(64,) dtype=torch.float32 device=cpu
placeholder  ; L_fn_modules_ln2_parameters_weight_ ; out=shape=(64,) dtype=torch.float32 device=cpu
placeholder  ; L_fn_modules_ln2_parameters_bias_ ; out=shape=(64,) dtype=torch.float32 device=cpu
UNSUPPORTED          ; op=call_function target=layer_norm in_args=[l_args_0_:shape=(2, 128, 64) dtype=torch.float32 device=cpu, [64], l_fn_modules_ln1_parameters_weight_:shape=(64,) dtype=torch.float32 device=cpu, l_fn_modules_ln1_parameters_bias_:shape=(64,) dtype=torch.float32 device=cpu, 1e-05] in_kwargs={} out=shape=(2, 128, 64) dtype=torch.float32 device=cpu
SUPPORTED batched_matmul_asm ; op=call_function key=linear target=<built-in function linear> in_args=[y:shape=(2, 128, 64) dtype=torch.float32 device=cpu, l_fn_modules_fc1_parameters_weight_:shape=(256, 64) dtype=torch.float32 device=cpu, l_fn_modules_fc1_parameters_bias_:shape=(256,) dtype=torch.float32 device=cpu] in_kwargs={} out=shape=(2, 128, 256) dtype=torch.float32 device=cpu
SUPPORTED gelu_asm     ; op=call_function key=gelu target=<built-in function gelu> in_args=[linear:shape=(2, 128, 256) dtype=torch.float32 device=cpu] in_kwargs={approximate='none'} out=shape=(2, 128, 256) dtype=torch.float32 device=cpu
SUPPORTED batched_matmul_asm ; op=call_function key=linear target=<built-in function linear> in_args=[gelu:shape=(2, 128, 256) dtype=torch.float32 device=cpu, l_fn_modules_fc2_parameters_weight_:shape=(64, 256) dtype=torch.float32 device=cpu, l_fn_modules_fc2_parameters_bias_:shape=(64,) dtype=torch.float32 device=cpu] in_kwargs={} out=shape=(2, 128, 64) dtype=torch.float32 device=cpu
SUPPORTED elementwise_add_asm ; op=call_function key=add target=<built-in function add> in_args=[l_args_0_:shape=(2, 128, 64) dtype=torch.float32 device=cpu, y_1:shape=(2, 128, 64) dtype=torch.float32 device=cpu] in_kwargs={} out=shape=(2, 128, 64) dtype=torch.float32 device=cpu
UNSUPPORTED          ; op=call_function target=layer_norm in_args=[x:shape=(2, 128, 64) dtype=torch.float32 device=cpu, [64], l_fn_modules_ln2_parameters_weight_:shape=(64,) dtype=torch.float32 device=cpu, l_fn_modules_ln2_parameters_bias_:shape=(64,) dtype=torch.float32 device=cpu, 1e-05] in_kwargs={} out=shape=(2, 128, 64) dtype=torch.float32 device=cpu
SUPPORTED elementwise_add_asm ; op=call_function key=add target=<built-in function add> in_args=[x:shape=(2, 128, 64) dtype=torch.float32 device=cpu, z:shape=(2, 128, 64) dtype=torch.float32 device=cpu] in_kwargs={} out=shape=(2, 128, 64) dtype=torch.float32 device=cpu
output       ; output ; out=?

 ===== PLENA ASM DUMP =====
graph id: 140443856276704
placeholder  ; L_fn_modules_ln1_parameters_weight_ ; out=shape=(64,) dtype=torch.float32 device=cpu
placeholder  ; L_fn_modules_ln1_parameters_bias_ ; out=shape=(64,) dtype=torch.float32 device=cpu
placeholder  ; L_args_0_ ; out=shape=(2, 128, 64) dtype=torch.float32 device=cpu
placeholder  ; L_fn_modules_fc1_parameters_weight_ ; out=shape=(256, 64) dtype=torch.float32 device=cpu
placeholder  ; L_fn_modules_fc1_parameters_bias_ ; out=shape=(256,) dtype=torch.float32 device=cpu
placeholder  ; L_fn_modules_fc2_parameters_weight_ ; out=shape=(64, 256) dtype=torch.float32 device=cpu
placeholder  ; L_fn_modules_fc2_parameters_bias_ ; out=shape=(64,) dtype=torch.float32 device=cpu
placeholder  ; L_fn_modules_ln2_parameters_weight_ ; out=shape=(64,) dtype=torch.float32 device=cpu
placeholder  ; L_fn_modules_ln2_parameters_bias_ ; out=shape=(64,) dtype=torch.float32 device=cpu
UNSUPPORTED          ; op=call_function target=layer_norm in_args=[l_args_0_:shape=(2, 128, 64) dtype=torch.float32 device=cpu, [64], l_fn_modules_ln1_parameters_weight_:shape=(64,) dtype=torch.float32 device=cpu, l_fn_modules_ln1_parameters_bias_:shape=(64,) dtype=torch.float32 device=cpu, 1e-05] in_kwargs={} out=shape=(2, 128, 64) dtype=torch.float32 device=cpu
SUPPORTED batched_matmul_asm ; op=call_function key=linear target=<built-in function linear> in_args=[y:shape=(2, 128, 64) dtype=torch.float32 device=cpu, l_fn_modules_fc1_parameters_weight_:shape=(256, 64) dtype=torch.float32 device=cpu, l_fn_modules_fc1_parameters_bias_:shape=(256,) dtype=torch.float32 device=cpu] in_kwargs={} out=shape=(2, 128, 256) dtype=torch.float32 device=cpu
SUPPORTED gelu_asm     ; op=call_function key=gelu target=<built-in function gelu> in_args=[linear:shape=(2, 128, 256) dtype=torch.float32 device=cpu] in_kwargs={approximate='none'} out=shape=(2, 128, 256) dtype=torch.float32 device=cpu
SUPPORTED batched_matmul_asm ; op=call_function key=linear target=<built-in function linear> in_args=[gelu:shape=(2, 128, 256) dtype=torch.float32 device=cpu, l_fn_modules_fc2_parameters_weight_:shape=(64, 256) dtype=torch.float32 device=cpu, l_fn_modules_fc2_parameters_bias_:shape=(64,) dtype=torch.float32 device=cpu] in_kwargs={} out=shape=(2, 128, 64) dtype=torch.float32 device=cpu
SUPPORTED elementwise_add_asm ; op=call_function key=add target=<built-in function add> in_args=[l_args_0_:shape=(2, 128, 64) dtype=torch.float32 device=cpu, y_1:shape=(2, 128, 64) dtype=torch.float32 device=cpu] in_kwargs={} out=shape=(2, 128, 64) dtype=torch.float32 device=cpu
UNSUPPORTED          ; op=call_function target=layer_norm in_args=[x:shape=(2, 128, 64) dtype=torch.float32 device=cpu, [64], l_fn_modules_ln2_parameters_weight_:shape=(64,) dtype=torch.float32 device=cpu, l_fn_modules_ln2_parameters_bias_:shape=(64,) dtype=torch.float32 device=cpu, 1e-05] in_kwargs={} out=shape=(2, 128, 64) dtype=torch.float32 device=cpu
SUPPORTED elementwise_add_asm ; op=call_function key=add target=<built-in function add> in_args=[x:shape=(2, 128, 64) dtype=torch.float32 device=cpu, z:shape=(2, 128, 64) dtype=torch.float32 device=cpu] in_kwargs={} out=shape=(2, 128, 64) dtype=torch.float32 device=cpu
output       ; output=[add_1:shape=(2, 128, 64) dtype=torch.float32 device=cpu]
