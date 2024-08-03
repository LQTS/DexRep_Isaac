from pathlib import Path
import os
import yaml

dir_path = Path("/remote-home/share/lqt/grasp_contactmap14/grasp_envs/grasp_suite/xml/assets_dexgraspnet0.5_szn_new_eval")
check_dir_path = Path("/remote-home/liuqingtao/UniDexGrasp/dexgrasp_policy/assets/meshdatav3_scaled")

files = dir_path.glob("Grasp_*")
files_names = dict()
for file in (files):
    full_name = file.stem[6:]
    ds_type = full_name.split('-')[0].lower()
    obj_name = full_name[len(ds_type)+1:]
    file_dir = check_dir_path / ds_type
    for f in file_dir.glob('*'):
        if obj_name == f.parts[-1].lower():
            obj_name = f.parts[-1]
            break
    files_names['<'+ds_type+'/'+obj_name+'<'] = "[0.1]>"

    file_path = file_dir / obj_name
    if not file_path.exists():
        print(file_path)
        continue
    # 确认是否可以找得到这条路径

# 将数据转换为YAML格式的字符串
yaml_str = yaml.dump(files_names)

# 将YAML格式的字符串写入文件
with open(f'eval_dexgraspnet_{len(files_names)}.yaml', 'w') as file:
    file.write(yaml_str)

print('YAML data has been written to data.yaml')