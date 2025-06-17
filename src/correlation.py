from extract_data import concatenate_df

df = concatenate_df()
tt = df['time_trace'][0]
print(tt.shape)
print(df.head())