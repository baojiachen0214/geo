## Python地理信息处理和可视化工具

### 项目简介
本项目旨在提供一套完整的地理信息处理和可视化工具，包括城市查找、文件URL获取、地理信息查询、数据匹配以及地理分布可视化等功能。通过调用中国行政区划的JSON API和本地地理数据表，可以实现对全球及中国地区地理信息的高效处理与展示。

### 文件结构
- **geo.py**: 包含所有核心功能函数，如城市查找、文件URL获取、地理信息查询、数据匹配等。
- **demo.ipynb**: 示例笔记本文件，演示如何使用`geo.py`中的函数进行数据处理和可视化。

### 功能介绍

#### 1. 城市查找 (`find_city`)
根据目标城市名称，在给定的地理数据中查找匹配的城市路径。支持自定义置信度阈值、返回路径的最大深度等参数。

#### 2. 文件URL获取 (`find_file`)
根据目标名称，从层级化的文件信息中查找对应的文件URL。

#### 3. 地理信息查询 (`search_geo`)
增强版地理信息查询函数，支持模糊匹配。可以根据不同的查询类型（英文名、中文名或ISO代码）进行精确或模糊搜索，并返回按相似度排序的结果列表。

#### 4. 数据清洗 (`normalize_chinese`)
针对中文字符的专用清洗函数，能够处理NaN值、繁体转简体、去除空格标点符号等操作。

#### 5. 匹配逻辑 (`get_gray_list_chinese`)
用于处理全局列表与输入列表之间的匹配关系，输出未匹配项、已匹配项及其余项。

#### 6. 批量匹配 (`match_data`)
批量匹配地理名称并分离匹配/未匹配的数据，适用于DataFrame格式的数据集。支持多种查询方式（英文名、中文名或ISO代码），并可选择是否启用模糊匹配。

#### 7. 地理分布可视化 (`plot_geo`)
基于Cartopy库绘制地理分布图，支持自定义颜色映射、图例配置等。可以将指定国家显示为灰色背景，并根据提供的数据列进行染色。

### 使用示例

在`demo.ipynb`中提供了详细的使用示例：

```python
# 导入模块
import geo
import pandas as pd

# 读取数据
df = pd.read_excel('./data.xlsx')

# 批量匹配地理名称
matched_df, unmatched_df = geo.match_data(find_style='en', dataframe=df, column_country='Name', column_data='data')

# 添加新行
matched_df.loc[len(matched_df)] = {'name': '英国', 'data': 14}

# 获取唯一国家列表
key_list = matched_df['name'].drop_duplicates().tolist()

# 获取未匹配、已匹配及剩余项
stack, popped_list, unmatched_list = geo.get_gray_list_chinese(key_list)

# 绘制地理分布图
geo.plot_geo(dataframe=matched_df, gray_countries=stack, column_data="data")
```


### 依赖库
- `requests`: 用于发起HTTP请求获取远程数据。
- `pandas`: 用于数据处理和分析。
- `geopandas`: 用于地理空间数据分析。
- `matplotlib`: 用于绘图。
- `cartopy`: 用于绘制地图。
- `zhconv`: 用于中文简繁转换。
- `difflib`: 用于字符串相似度比较。

### 安装方法
确保已安装上述依赖库，可以通过以下命令安装：
```bash
pip install requests pandas geopandas matplotlib cartopy zhconv
```


### 注意事项
- 在使用`plot_geo`函数时，请确保GeoJSON文件路径正确无误。
- 对于模糊匹配功能，建议根据实际需求调整相似度阈值以获得最佳匹配效果。

---
希望能够帮助您更好地理解和使用本项目。如果有任何问题或建议，欢迎随时联系！
