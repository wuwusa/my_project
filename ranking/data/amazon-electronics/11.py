import pandas as pd

df3 = pd.read_csv('data03.csv')
# # 读取第一个CSV文件
# df1 = pd.read_csv('data01.csv')
#
# # 读取第二个CSV文件
# df2 = pd.read_csv('data02.csv')
#
# # 将df2中的upload_type列与df1合并，基于共有的video_id列
# merged_df = df1.merge(df2[['video_id', 'music_type','upload_type']], left_on='video_id', right_on='video_id', how='left')
#
# # 保存合并后的结果到新的CSV文件
# merged_df.to_csv('data03.csv', index=False)
#
# df3 = pd.read_csv('data03.csv')
# df3['time_ms']=df3['time_ms'] // 1000   # 毫秒级时间戳转换成秒级时间戳
# df3.rename(columns={'time_ms': 'time'}, inplace=True)
# df3.rename(columns={'upload_type': 'cate'}, inplace=True)
#
# 使用 factorize 方法将分类数据映射为数字，并获取映射后的整数编码数组
active_degree_id = pd.factorize(df3['user_active_degree'])[0]
df3['active_degree_id']=active_degree_id+1
df3 = df3.drop('user_active_degree', axis=1)
# df3['music_type'].fillna(1, inplace=True)
df3.to_csv('data03.csv',index=False)
# -------------------从user_feature_pure中获得------------------------------
# 读取目标文件和源文件
# target_df = pd.read_csv('data03.csv')
# source_df = pd.read_csv('user_features_pure.csv')

# 假设你要添加的列名为 'column_to_add'，连接键列名为 'common_column'
# column_to_add = 'user_active_degree'
# common_column = 'user_id'

# 将源文件中的指定列添加到目标文件中，基于连接键
# target_df = target_df.merge(source_df[[common_column, column_to_add]], on=common_column, how='left')

# 保存修改后的结果回到目标文件
# target_df.to_csv('data03.csv', index=False)
# ---------------------------------------------------------------

print(df3['active_degree_id'])