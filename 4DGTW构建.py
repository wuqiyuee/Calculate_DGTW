import  pandas as pd
import numpy as np 
import datetime as dt 
import matplotlib.pyplot as plt 
from dateutil.relativedelta import * 
from pandas.tseries.offsets import * 
import warnings
warnings.filterwarnings('ignore') 
pd.options.mode.chained_assignment = None 
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
pd.set_option('display.max_columns', None)

#读取日度股价的数据，用于提取市值并排名、计算动量
df_total = pd.read_csv(
    '/Users/qiyuewu/Desktop/mmm复现数据包/下载数据/CSMR/个股日度收益率/CSMAR_日个股回报率_2000_2025.csv',
    encoding='utf-8')
mom = df_total.loc[(df_total['Markettype'].isin([1,4]))].copy()
mom['Trddt'] = pd.to_datetime(mom['Trddt'])
mom['jdate'] = mom['Trddt'] + MonthEnd(0)
mom['size'] = mom['Dsmvtll'] * 1000  
mom['sizew'] = mom['Dsmvosd'] * 1000  
mom=mom.rename(columns={'Dretwd':'ret'})
mom=mom[['Stkcd','Trddt','jdate','sizew','size','ret','Trdsta']]

################################################################################
########################### 处理日度文件，计算动量 mom #############################
################################################################################

na_count = mom['ret'].isna().sum()#为0

#计算指数
mom['prcIndex']=mom.groupby('Stkcd')['ret'].transform(lambda x:(1+x).cumprod())
#计算return_tm252_tm20
mom['idx_t20'] = mom.groupby('Stkcd')['prcIndex'].shift(20)
mom['idx_t252'] = mom.groupby('Stkcd')['prcIndex'].shift(252)
mom['return_tm252_tm20'] = (mom['idx_t20'] / mom['idx_t252'] - 1) * 100
mom.drop(columns=['idx_t20', 'idx_t252'], inplace=True)
mom=mom[['Stkcd','Trddt','jdate','sizew','size','ret','return_tm252_tm20','Trdsta']]

################################################################################
########################从日度交易数据中提取月末市值并进行分组Size####################
################################################################################

# 提取每个股票每月的最后一个交易日记录，生成TRD_Mnth
TRD_Mnth = mom.sort_values(['Stkcd', 'Trddt']).groupby(['Stkcd', 'jdate']).tail(1).copy()
#选择变量，这里带上之前计算的return_tm252_tm20，相当于直接合并了
TRD_Mnth = TRD_Mnth[['Stkcd','Trddt','jdate', 'size','sizew', 'ret','return_tm252_tm20','Trdsta']]

###计算市值断点
TRD_Mnth=TRD_Mnth.sort_values(['jdate','Stkcd']).drop_duplicates()
bk=TRD_Mnth.groupby('jdate')['sizew'].describe(percentiles=[.2,.4,.6,.8])\
.reset_index()
bk=bk[['jdate','20%','40%','60%','80%']].rename(
    columns={
    '20%':'dec20',
    '40%':'dec40',
    '60%':'dec60',''
    '80%':'dec80'})
size=pd.merge(TRD_Mnth,bk,how='left',on='jdate')
#计算市值分组
def size_group(row):
    if 0<=row['sizew']<row['dec20']:
        value=5
    elif row['sizew']<row['dec40']:
        value=4
    elif row['sizew']<row['dec60']:
        value=3
    elif row['sizew']<row['dec80']:
        value=2
    elif row['sizew']>=row['dec80']:
        value=1
    else:
        value=np.nan
    return value
size['group']=size.apply(size_group,axis=1)
size['year_prv']=size['jdate'].dt.year-1
size_mom=size[['Stkcd','Trddt','jdate','year_prv','group','size','sizew','return_tm252_tm20']]

################################################################################
######读取资产负债表的数据（季度数据），计算账面权益（上年12月底），合并市值数据后计算bm#######
################################################################################
FS_Combas = pd.read_csv(
    '/Users/qiyuewu/Desktop/mmm复现数据包/下载数据/CSMR/资产负债表2000-01-01至2025-12-31/FS_Combas.csv',
                    encoding='utf-8')
comp=FS_Combas.copy()
comp=comp.loc[comp['Typrep']=='A']
comp=comp.rename(
    columns={
        'Accper':'date',
        'A001222000':'递延所得税资产',
        'A002208000':'递延所得税负债',
        'A003112000':'oth_eq_inst', 
        'A003100000':'eqToParent',
        'A003000000':'teq',
        'A003112101':'pref'})
comp['date']=pd.to_datetime(comp['date'])
comp=comp.loc[comp['date'].dt.month==12]#原表为季度数据，转换为年度数据
comp['year']=comp['date'].dt.year
comp['jdate']=comp['date']+MonthEnd(0)
comp=comp[['Stkcd','date','Typrep','oth_eq_inst','eqToParent','teq','year','jdate','递延所得税负债','递延所得税资产','pref']]

#优先使用归属于母公司的所有者权益计算，没有的话使用总权益，都没有填充为0
comp['eq']=np.where(comp['eqToParent'].isnull(),comp['teq'],comp['eqToParent'])
comp['eq']=np.where(comp['eq'].isnull(),0,comp['eq'])
comp['oth_eq_inst']=comp['oth_eq_inst'].fillna(0)
comp['pref'] = comp['pref'].fillna(0)
comp['递延所得税负债'] = comp['递延所得税负债'].fillna(0)
comp['递延所得税资产'] = comp['递延所得税资产'].fillna(0)
comp['be']=comp['eq']-comp['pref']+comp['递延所得税负债']-comp['递延所得税资产']#计算所有者账面价值
comp=comp.loc[comp['be']>=0]#筛选出账面价值大于0的
comp=comp[['Stkcd','date','Typrep','jdate','year','be']]

#合并市值和账面市值,计算账面市值比
comp1=pd.merge(comp,size_mom,how='inner',on=['Stkcd','jdate'])
comp1['bm']=np.where(comp1['sizew']>0,comp1['be']/comp1['sizew'],np.nan)
average_bm = comp1['bm'].mean()

# ==============================================================================
# ============================== 进行二级行业调整 ================================
# ==============================================================================

#读取上市公司行业分类年度表
ind_path = '/Users/qiyuewu/Desktop/mmm复现数据包/下载数据/CSMR/DGTW/DGTW股票分配表/STK_MKT_DGTWASSINGMENTS.csv'
ind_df = pd.read_csv(ind_path, encoding='utf-8')

#数据预处理
ind_df = ind_df.rename(columns={'Symbol': 'Stkcd'})
#转换日期并提取年份，用于后续与财务数据(comp1)匹配
ind_df['TradingYear'] = ind_df['TradingYear'].astype(int)
ind=ind_df.loc[ind_df['IsNotBSE']==1]
ind=ind[['Stkcd', 'TradingYear', 'IndustryCode']]

ind['year']=ind['TradingYear']-1
comp1 = pd.merge(comp1, ind, on=['Stkcd', 'year'], how='left')

ind_year_bm=(
    comp1.dropna(subset=['IndustryCode','bm'])
    .groupby(['IndustryCode','year'],as_index=False)
    .agg(bmind=('bm','mean'))
    .sort_values(['IndustryCode','year'])
)
#计算行业累计的平均值
ind_year_bm['bmavg']=(
    ind_year_bm.groupby('IndustryCode')['bmind']
    .transform(lambda x:x.expanding().mean())
)

comp1=pd.merge(
    comp1,
    ind_year_bm[['IndustryCode','year','bmind','bmavg']],
    on=['IndustryCode','year'],
    how='left'
)

comp1['adj_bm'] = comp1['bm'] -comp1['bmavg']

print(f"已完成基于 STK_INDUSTRYCLASSANL二级行业的 AdjBM 计算")
print(f"   匹配样本数: {len(comp1)}, 二级行业缺失数: {comp1['IndustryCode'].isna().sum()}")
#行业缺失的adj_bm会是缺失值，在后续分组前会被删除

################################################################################
######### 将月度的市值——动量表、年度的账面价值以及账面市值比进行合并（月度层面）#########
################################################################################

#仅筛选出6月份的数据
size_mom6= size_mom[size_mom['jdate'].dt.month==6].copy()

#合并comp1得到上年末的bm数据
bm=comp1[['Stkcd','year','bm','adj_bm']]
bm=bm.rename(columns={'year':'Connect_year'})
size_mom6=size_mom6.rename(columns={'year_prv':'Connect_year'})
size_mom_bm=pd.merge(size_mom6,bm,how='inner',on=['Stkcd','Connect_year'])
size_mom_bm=size_mom_bm.dropna(subset=['size','return_tm252_tm20','adj_bm','bm'],how='any')

################################################################################
########################进行三重排序，构建DGTW组合##################################
################################################################################

# 开始进行三重排序：市值、账面市值比、动量
port1=size_mom_bm.sort_values(['jdate','group','Stkcd']).drop_duplicates()

port1['bmr']=port1.groupby(['jdate','group'])['adj_bm'].\
    transform(lambda x:pd.qcut(x,5,labels=False))
port2=port1.sort_values(['jdate','group','bmr'])
port2['momr']=port2.groupby(['jdate','group','bmr'])['return_tm252_tm20'].\
    transform(lambda x: pd.qcut(x,5,labels=False))

#1表示最大，5表示最小（为了符合CSMAR的习惯）
port3=port2.copy()
port3 = port3.dropna(subset=['group', 'bmr', 'momr'])
port3['group'] = port3['group'].astype(int)
port3['bmr'] = 5 - port3['bmr'].astype(int)
port3['momr'] = 5 - port3['momr'].astype(int)
port3[['group','bmr','momr']]=port3[['group','bmr','momr']].astype(int).astype(str)
port3['dgtw_port']=port3['group']+port3['bmr']+port3['momr']
port4=port3[['Stkcd','jdate','sizew','return_tm252_tm20','bm','dgtw_port']]
port4['jyear']=port4['jdate'].dt.year
port4=port4.sort_values(['Stkcd','jdate'])
port4=port4.rename(columns={'jdate':'formdate',})
port4=port4[['Stkcd','formdate','jyear','sizew','dgtw_port']]
print(f"构建的DGTW组合：\n{port4.head()}")

################################################################################
##################################计算基准收益率##################################
################################################################################

#将组合分配回日度收益文件
mom=mom.sort_values(['Stkcd','Trddt'])
#权重为t-3的市值
mom['weight_t3']=mom.groupby('Stkcd')['sizew'].shift(3)
mom['month'] = mom['Trddt'].dt.month
mom['jyear'] = np.where(
    mom['month'] <= 6, 
    mom['Trddt'].dt.year - 1, 
    mom['Trddt'].dt.year
)
# 仅保留计算必要的列，节省内存
mom_daily_clean = mom[['Stkcd', 'Trddt', 'weight_t3', 'jyear','ret']].copy()

#合并标签
#确保 port4 中的 Stkcd 和 jyear 与 mom 中的类型一致
dgtw_daily = pd.merge(
    mom_daily_clean, 
    port4[['Stkcd', 'jyear', 'dgtw_port']], 
    on=['Stkcd', 'jyear'], 
    how='left'
)

#剔除无法匹配标签或权重缺失的行
dgtw_daily = dgtw_daily.dropna(subset=['dgtw_port', 'weight_t3'])
#定义加权平均函数
def wavg(group, avg_name, weight_name):
    # 仅保留收益率和权重都不为缺失值的干净样本
    valid = group.dropna(subset=[avg_name, weight_name])
    if len(valid) == 0 or valid[weight_name].sum() == 0:
        return np.nan
    
    d = group[avg_name]
    w = group[weight_name]
    return (d * w).sum() / w.sum()

#计算 125 个组合每日的基准收益率
print("正在计算每日 125 个组合的加权平均收益...")
daily_vwr = dgtw_daily.groupby(['Trddt', 'dgtw_port']).apply(
    wavg, 'ret', 'weight_t3'
).reset_index().rename(columns={0: 'benchmark_ret'})

#合并回主表并计算日度超额收益 (Excess Return)
final_daily = pd.merge(
    dgtw_daily, 
    daily_vwr, 
    on=['Trddt', 'dgtw_port'], 
    how='left'
)

final_daily['dgtw_daily_xret'] = final_daily['ret'] - final_daily['benchmark_ret']

print("日度 DGTW 超额收益计算完成！")
print(final_daily[['Stkcd', 'Trddt', 'dgtw_port', 'benchmark_ret', 'dgtw_daily_xret']].head())

stats_return = final_daily['dgtw_daily_xret'].describe()
print("\ndgtw_daily_xret的统计：")
print(stats_return)



# =====================================================================
# 终极检验模块 (请放在 final_daily 计算完成之后)
# =====================================================================

print("\n" + "="*50)
print("检验一：月度基准收益率 (Benchmark Returns) 相关性测试")
print("="*50)

# 1. 使用 daily_vwr 计算你自己的月度组合基准收益
daily_vwr = daily_vwr.copy()
daily_vwr['Trddt'] = pd.to_datetime(daily_vwr['Trddt'])
daily_vwr['TradingMonth'] = daily_vwr['Trddt'].dt.to_period('M')

# 累乘得到月度收益
my_monthly_bench = daily_vwr.groupby(['TradingMonth', 'dgtw_port'])['benchmark_ret'].apply(
    lambda x: (1 + x.dropna()).prod() - 1
).reset_index(name='my_bench_ret')
my_monthly_bench['dgtw_port'] = my_monthly_bench['dgtw_port'].astype(int)

# 2. 读取官方月度基准收益表
STK_MKT_DGTWBENCH = pd.read_csv('/Users/qiyuewu/Desktop/mmm复现数据包/下载数据/CSMR/DGTW/DGTW特征基准指标/STK_MKT_DGTWBENCH.csv', encoding='utf-8')
STK_MKT_DGTWBENCH = STK_MKT_DGTWBENCH[
    STK_MKT_DGTWBENCH['IsNotBSE'] == 1
].copy()
# 官方标签合成 111-555
STK_MKT_DGTWBENCH['dgtw_port'] = (STK_MKT_DGTWBENCH['MarketValue'].astype(int).astype(str) + 
                                  STK_MKT_DGTWBENCH['BooktoMarket'].astype(int).astype(str) + 
                                  STK_MKT_DGTWBENCH['Momentum'].astype(int).astype(str)).astype(int)
STK_MKT_DGTWBENCH['TradingMonth'] = pd.to_datetime(STK_MKT_DGTWBENCH['TradingMonth']).dt.to_period('M')

# 3. 合并计算相关系数
corr_bench = pd.merge(my_monthly_bench, STK_MKT_DGTWBENCH, how='inner', on=['TradingMonth', 'dgtw_port'])

if not corr_bench.empty:
    correlation = corr_bench['my_bench_ret'].corr(corr_bench['BenchmarkReturns'])
    print(f"✅ 月度 Benchmark 收益率相关系数: {correlation:.4f} ")
else:
    print("❌ 基准收益合并失败，请检查数据。")

print("\n" + "="*50)
print("检验二：年度个股 DGTW 标签 (Assignments) 精确匹配率测试")
print("="*50)

# 1. 提取你的 6 月份分组快照 (port4)
port4_check = port4.copy()
port4_check['dgtw_port'] = port4_check['dgtw_port'].astype(int)
port4_check['TradingYear'] = port4_check['jyear'].astype(int)

# 拆解你自己的 1-5 标签 (1最大, 5最小)
port4_check['my_size'] = port4_check['dgtw_port'] // 100
port4_check['my_bm'] = (port4_check['dgtw_port'] // 10) % 10
port4_check['my_mom'] = port4_check['dgtw_port'] % 10

# 2. 读取官方的股票分配表
STK_MKT_DGTWASSINGMENTS = pd.read_csv('/Users/qiyuewu/Desktop/mmm复现数据包/下载数据/CSMR/DGTW/DGTW股票分配表/STK_MKT_DGTWASSINGMENTS.csv', encoding='utf-8')
STK_MKT_DGTWASSINGMENTS = STK_MKT_DGTWASSINGMENTS[
    STK_MKT_DGTWASSINGMENTS['IsNotBSE'] == 1
].copy()
STK_MKT_DGTWASSINGMENTS['Stkcd'] = STK_MKT_DGTWASSINGMENTS['Symbol'].astype(int)
STK_MKT_DGTWASSINGMENTS['TradingYear'] = STK_MKT_DGTWASSINGMENTS['TradingYear'].astype(int)

# 合成官方整体 3 位数标签
STK_MKT_DGTWASSINGMENTS['off_dgtw'] = (STK_MKT_DGTWASSINGMENTS['MarketValue'].astype(int).astype(str) + 
                                       STK_MKT_DGTWASSINGMENTS['BooktoMarket'].astype(int).astype(str) + 
                                       STK_MKT_DGTWASSINGMENTS['Momentum'].astype(int).astype(str)).astype(int)

# 严格按【股票】和【年份(TradingYear)】进行内连接对齐
match_df = pd.merge(
    port4_check[['Stkcd', 'TradingYear', 'dgtw_port', 'my_size', 'my_bm', 'my_mom']],
    STK_MKT_DGTWASSINGMENTS[['Stkcd', 'TradingYear', 'off_dgtw', 'MarketValue', 'BooktoMarket', 'Momentum']],
    on=['Stkcd', 'TradingYear'], 
    how='inner'
)

# 4. 统计匹配率
total_samples = len(match_df)

if total_samples > 0:
    exact_match = (match_df['dgtw_port'] == match_df['off_dgtw']).mean() * 100
    size_match = (match_df['my_size'] == match_df['MarketValue']).mean() * 100
    bm_match = (match_df['my_bm'] == match_df['BooktoMarket']).mean() * 100
    mom_match = (match_df['my_mom'] == match_df['Momentum']).mean() * 100

    print(f"参与比对的样本总数 (精确到股票-年份): {total_samples}")
    print(f"✅ 三维完全相等的准确率: {exact_match:.2f}%")
    print("-" * 30)
    print(f"🔍 细分维度匹配率诊断：")
    print(f"Size (规模) 匹配率: {size_match:.2f}%")
    print(f"BM (账面市值比) 匹配率: {bm_match:.2f}%")
    print(f"Momentum (动量) 匹配率: {mom_match:.2f}%")
else:
    print("❌ 标签对齐失败：未匹配到相同的股票和年份，请检查 jyear 的对齐逻辑。")


# =====================================================================
# 附加模块：差集提取与错位分析 (Mismatch Analysis)
# =====================================================================
print("\n" + "="*50)
print("开始提取并分析标签不一致的差异样本...")
print("="*50)

# 1. 筛选出你自己的标签与官方标签不相等的行
mismatch_df = match_df[match_df['dgtw_port'] != match_df['off_dgtw']].copy()

if len(mismatch_df) == 0:
    print("🎉 太棒了！你的分组和官方 100% 完全一致，无需排查！")
else:
    print(f"⚠️ 发现 {len(mismatch_df)} 条记录的组合标签与官方存在差异。")
    print(f"占总匹配样本的比例为: {len(mismatch_df) / total_samples * 100:.2f}%\n")

    # 2. 计算每个维度的偏差幅度 (你的组别 - 官方组别)
    # 正数说明你分到了更大的组(比如你分在组5，官方在组4)，负数反之
    mismatch_df['diff_size'] = mismatch_df['my_size'] - mismatch_df['MarketValue']
    mismatch_df['diff_bm'] = mismatch_df['my_bm'] - mismatch_df['BooktoMarket']
    mismatch_df['diff_mom'] = mismatch_df['my_mom'] - mismatch_df['Momentum']

    # 3. 统计一下错位程度（只错位了1组，还是错位了2组以上？）
    # 使用绝对值来衡量错位幅度
    for col, name in zip(['diff_size', 'diff_bm', 'diff_mom'], ['Size', 'BM', 'Momentum']):
        off_by_1 = (mismatch_df[col].abs() == 1).sum()
        off_by_more = (mismatch_df[col].abs() > 1).sum()
        total_mismatch = off_by_1 + off_by_more
        
        if total_mismatch > 0:
            print(f"【{name} 维度错位分析】总计差异数: {total_mismatch}")
            print(f"   -> 仅错位 1 个组别 (通常是处在断点边缘): {off_by_1} ({off_by_1/total_mismatch*100:.1f}%)")
            print(f"   -> 错位 2 个及以上组别 (需重点排查基础数据): {off_by_more} ({off_by_more/total_mismatch*100:.1f}%)")
    
    # 4. 把相关的原始变量（市值、收益率等）也合并进来，方便归因
    # 假设之前的 port4 和 off_val 在内存中还可用，我们可以把计算断点的基础值带上
    mismatch_detail = pd.merge(
        mismatch_df,
        port4_check[['Stkcd', 'TradingYear', 'sizew']], # 你的实际市值
        on=['Stkcd', 'TradingYear'],
        how='left'
    )
    
    # 将结果按年份和股票代码排序，方便查阅
    mismatch_detail = mismatch_detail.sort_values(['TradingYear', 'Stkcd'])
    
    # 5. 导出为 CSV 文件供 Excel 分析
    output_filename = 'DGTW_Mismatch_Analysis.csv'
    mismatch_detail.to_csv(output_filename, index=False, encoding='utf-8-sig') # utf-8-sig 确保 Excel 打开中文不乱码
    print(f"\n✅ 详细的差异对比明细已导出至当前目录: {output_filename}")
    
    # 6. 在控制台预览前 5 条典型的差异数据
    print("\n--- 差异数据预览 (Top 5) ---")
    preview_cols = ['Stkcd', 'TradingYear', 'dgtw_port', 'off_dgtw', 'diff_size', 'diff_bm', 'diff_mom']
    print(mismatch_detail[preview_cols].head())


print("\n" + "="*50)
print("排除级联误差：只看 Size 完美匹配的股票")
print("="*50)

# 修复：先在全量数据 match_df 上计算好所有的 diff 列
match_df['diff_size'] = match_df['my_size'] - match_df['MarketValue']
match_df['diff_bm'] = match_df['my_bm'] - match_df['BooktoMarket']
match_df['diff_mom'] = match_df['my_mom'] - match_df['Momentum']

# 筛选出你和官方 Size 标签完全一致的样本 (diff_size == 0)
size_matched_df = match_df[match_df['diff_size'] == 0]

print(f"Size 完全匹配的样本总数: {len(size_matched_df)}")

# 在这些 Size 匹配的样本中，看 BM 有多少错位
bm_mismatch_in_size = size_matched_df[size_matched_df['diff_bm'] != 0]
print(f"其中，BM 依然出现错位的样本数: {len(bm_mismatch_in_size)} (占 Size 匹配样本的 {len(bm_mismatch_in_size)/len(size_matched_df)*100:.2f}%)")

# 进一步：看 Size 和 BM 都完全匹配的样本中，Momentum 有多少错位
size_bm_matched_df = size_matched_df[size_matched_df['diff_bm'] == 0]
mom_mismatch_in_size_bm = size_bm_matched_df[size_bm_matched_df['diff_mom'] != 0]
print(f"其中，动量 依然出现错位的样本数: {len(mom_mismatch_in_size_bm)} (占 Size&BM 均匹配样本的 {len(mom_mismatch_in_size_bm)/len(size_bm_matched_df)*100:.2f}%)")
