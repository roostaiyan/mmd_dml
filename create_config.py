import os

template_path = 'config.ini'

datasets = []
#name, cat_cnt, train_size, pair_cnt
datasets.append(('tiny', '10', '410', '1640'))

# depth of modality-specific SAEs
depth = ['2']

# depth of JSAE
jdepth = '2'
# whether to use JSAE, output width of JSAE
joint = [('True', '64')]

f = open(template_path,'r')
template = f.read()
f.close()

for name, cat_cnt, train_size, pair_cnt in datasets:
    print name
    save_dir = os.path.join('config',name)
    ind = 0
    for has_joint, final_dim in joint:
        for d in depth:
            filedata = template
            filedata = filedata.replace('*dataset_name*', name)
            filedata = filedata.replace('*cat_cnt*', cat_cnt)
            filedata = filedata.replace('*train_size*', train_size)
            filedata = filedata.replace('*pair_cnt*', pair_cnt)
            filedata = filedata.replace('*depth*', d)
            filedata = filedata.replace('*has_joint*', has_joint)
            filedata = filedata.replace('*final_dim*', final_dim)
            filedata = filedata.replace('*jdepth*', jdepth)
            filedata = filedata.replace('*ind*', str(ind))
#             print filedata

            f = open(os.path.join(save_dir, 'config'+str(ind)+'.ini'),'w')
            f.write(filedata)
            f.close()
            ind += 1
