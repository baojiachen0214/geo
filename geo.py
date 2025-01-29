import requests
import pandas as pd
from typing import Tuple
from difflib import get_close_matches
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import re
import zhconv  # 需要安装：pip install zhconv

CHINA_URL = "https://geojson.cn/api/china/_meta.json"
GEO_LIST = pd.read_excel('./geo/list.xls')
GEO_COUNTRY_LIST = GEO_LIST['名称']


def connect_json(url):
    """
    从给定的URL获取JSON数据。

    发起GET请求到指定的URL地址，如果响应的状态码是200（即请求成功），
    则将响应内容解析为JSON格式的数据并返回；否则，打印出状态码并返回None。

    参数:
    url (str): 用于GET请求的URL地址。

    返回:
    dict: 解析后的JSON数据。
    如果响应状态码不是200，则返回None，并打印出状态码。
    """
    # 发起GET请求以获取数据
    response = requests.get(url)
    # 检查响应状态码
    if response.status_code == 200:
        # 如果状态码为200，返回解析后的JSON数据
        return response.json()
    else:
        # 如果状态码非200，打印状态码并返回None
        print(f"状态码: {response.status_code}")
        return None


def find_city(target, conf=1.0, data=connect_json(CHINA_URL), path="", depth=None):
    """
    在给定的数据中查找与目标城市名匹配的城市路径。

    参数:
    - target: 目标城市名字符串。
    - conf: 匹配置信度阈值，默认为1.0，表示完全匹配。取值范围为 (0,1)
    - data: 地理数据，默认为从CHINA_URL获取的中国地理数据。
    - path: 当前搜索路径，默认为空字符串。
    - depth: 返回路径的最大深度，默认为None，表示返回所有匹配的路径。

    返回:
    - 匹配的城市路径列表。
    """
    # 存储匹配的城市路径
    matches = []

    def lcs(X, Y):
        """
        计算两个字符串的最长公共子序列（LCS）长度。

        参数:
        - X: 第一个字符串。
        - Y: 第二个字符串。

        返回:
        - 最长公共子序列的长度。
        """
        m = len(X)
        n = len(Y)
        # 创建一个 (m+1) x (n+1) 的二维数组来存储子问题的解
        L = [[0] * (n + 1) for _ in range(m + 1)]

        # 填充 L 数组
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 or j == 0:
                    L[i][j] = 0
                elif X[i - 1] == Y[j - 1]:
                    L[i][j] = L[i - 1][j - 1] + 1
                else:
                    L[i][j] = max(L[i - 1][j], L[i][j - 1])

        # L[m][n] 包含 LCS 的长度
        return L[m][n]

    def calculate_match_score(name, target):
        """
        计算城市名与目标字符串的匹配分数。

        参数:
        - name: 城市名字符串。
        - target: 目标城市名字符串。

        返回:
        - 匹配分数，表示两个字符串的相似程度。
        """
        # 边界条件检查
        if not isinstance(name, str) or not isinstance(target, str):
            raise ValueError("Both 'name' and 'target' must be strings.")

        if not name or not target:
            return 0

        # 使用最长公共子序列计算匹配分数
        return lcs(name, target)

    def search(node, current_path, current_depth):
        """
        在地理数据树中递归搜索匹配的城市路径。

        参数:
        - node: 当前节点的数据。
        - current_path: 当前搜索路径。
        - current_depth: 当前搜索深度。
        """
        # 更新当前搜索路径
        current_path = current_path + " " + node['name'] if current_path else node['name']
        # 计算当前节点与目标城市的匹配分数
        match_score = calculate_match_score(node['name'], target)

        # 归一化匹配分数
        normalized_score = match_score / max(len(node['name']), len(target))

        # 如果归一化后的匹配分数达到或超过置信度阈值，则将当前路径添加到匹配列表中
        if normalized_score >= conf:
            matches.append(current_path)

        # 如果当前节点有子节点且未达到最大深度，则递归搜索子节点
        if 'children' in node and (depth is None or current_depth < depth):
            for child in node['children']:
                search(child, current_path, current_depth + 1)

    def expand_paths(paths, data, depth):
        """
        扩展路径以包含子路径。

        参数:
        - paths: 当前匹配的路径列表。
        - data: 地理数据。
        - depth: 返回路径的最大深度。

        返回:
        - 扩展后的路径列表。
        """
        expanded_paths = []

        def expand(node, current_path, current_depth, visited):
            """
            递归扩展路径。

            参数:
            - node: 当前节点的数据。
            - current_path: 当前搜索路径。
            - current_depth: 当前搜索深度。
            - visited: 已访问的节点名称集合。
            """
            # 更新当前搜索路径
            if node['name'] not in visited:
                current_path = current_path + " " + node['name'] if current_path else node['name']
                visited.add(node['name'])
            # 如果当前深度未达到最大深度，则将当前路径添加到扩展路径列表中
            if current_depth <= depth:
                expanded_paths.append(current_path)
            # 如果当前节点有子节点且未达到最大深度，则递归搜索子节点
            if 'children' in node and current_depth < depth:
                for child in node['children']:
                    expand(child, current_path, current_depth + 1, visited)

        # 遍历每个匹配的路径
        for path in paths:
            # 找到匹配路径的最后一个节点
            path_parts = path.split(" ")
            current_node = data
            for part in path_parts:
                if 'children' in current_node:
                    for child in current_node['children']:
                        if child['name'] == part:
                            current_node = child
                            break

            # 从匹配路径的最后一个节点开始扩展
            expand(current_node, path, len(path_parts), set(path_parts))

        return expanded_paths

    # 从给定的地理数据开始搜索
    search(data, path, 0)

    # 如果 depth 不为 None，则扩展路径
    if depth is not None:
        matches = expand_paths(matches, data, depth)

    # 返回匹配的城市路径列表
    return matches


def find_file(target: str, data=connect_json(CHINA_URL)):
    """
    根据目标名称查找对应的文件URL。

    参数：
    - target: 目标名称，用于查找对应的文件。
    - data: 初始数据结构，包含层级化的文件信息，默认为连接到中国的JSON数据。

    返回：
    - 查找到的文件URL。
    """
    # 分割目标名称，以便逐级匹配
    names = target.split(" ")
    # 移除第一个元素，因为它不用于文件查找
    names.pop(0)
    # 获取需要匹配的名称数量
    n = len(names)

    # 遍历每个名称，逐级查找匹配的数据
    for i in range(n):
        # 遍历当前数据的子元素
        for child in data['children']:
            # 当找到匹配的子元素时，更新数据为该子元素，以便进行下一级匹配
            if child['name'] == names[i]:
                data = child
                break

    # 分割文件路径，获取最后一个元素作为文件名
    key = data['filename'].split("/").pop()
    # 拼接完整的文件URL
    filename = f"https://geo.datav.aliyun.com/areas_v3/bound/{key}.json"
    # 返回文件URL
    return filename


def search_geo(
        query_type: str,
        query_content: str,
        fuzzy: bool = False,
        cutoff: float = 0.6,
        limit: int = 5
) -> list:
    """
    增强版地理信息查询函数，支持模糊匹配

    :param query_type: 查询类型，可选'en'/'zh'/'ISO'
    :param query_content: 查询内容（大小写不敏感）
    :param fuzzy: 是否启用模糊匹配（默认关闭）
    :param cutoff: 模糊匹配相似度阈值（0-1，默认0.6）
    :param limit: 最大返回结果数（默认5）
    :return: 按相似度排序的中文名称列表，无匹配返回空列表
    """
    # 验证输入有效性
    if query_type not in ['en', 'zh', 'ISO']:
        raise ValueError(f"无效的查询类型: {query_type}，支持'en'/'zh'/'ISO'")

    # 定义查询映射关系
    column_map = {
        'en': ['Name', 'FName'],
        'zh': ['名称', '全称'],
        'ISO': ['ISO_A2', 'ISO_A3']
    }

    # 预处理输入内容
    query = query_content.strip().lower()

    # 精确匹配模式
    if not fuzzy:
        mask = pd.Series(False, index=GEO_LIST.index)
        for col in column_map[query_type]:
            mask |= (GEO_LIST[col].str.strip().str.lower() == query)
        results = GEO_LIST.loc[mask, '名称'].tolist()
        return results if results else []

    # 模糊匹配模式
    all_candidates = set()
    for col in column_map[query_type]:
        # 标准化列数据并生成候选列表
        candidates = GEO_LIST[col].str.strip().str.lower().dropna().unique()
        # 使用difflib获取近似匹配
        matches = get_close_matches(
            query,
            candidates,
            n=limit,
            cutoff=cutoff
        )
        # 获取原始大小写的匹配项
        for match in matches:
            original = GEO_LIST[GEO_LIST[col].str.strip().str.lower() == match]
            all_candidates.update(original['名称'].tolist())

    # 按数据原始顺序去重
    seen = set()
    ordered_results = []
    for name in GEO_LIST['名称']:
        if name in all_candidates and name not in seen:
            ordered_results.append(name)
            seen.add(name)
            if len(seen) >= limit:
                break

    return ordered_results


def normalize_chinese(s):
    """中文专用清洗函数：
       1. 处理 NaN
       2. 繁体转简体
       3. 去除所有空格、标点、特殊符号
       4. 统一为无符号格式
    """
    if pd.isna(s):
        return ""
    s = str(s)
    # 繁体转简体
    s = zhconv.convert(s, 'zh-hans')
    # 去除非中文字符、标点、空格
    s = re.sub(r'[^\u4e00-\u9fa5]', '', s)  # 仅保留汉字
    return s.lower()  # 可选：如果列表中有英文大小写混合可以统一


def get_gray_list_chinese(input_list):
    # 处理全局列表
    remaining_global = [item for item in GEO_COUNTRY_LIST]
    popped_list = []
    unmatched_list = []

    # 处理输入列表
    processed_input = [item for item in input_list]

    # 调试输出（确保控制台支持中文编码）
    print("[DEBUG] GEO_COUNTRY_LIST 处理后:", remaining_global)
    print("[DEBUG] 输入列表处理后:", processed_input)

    # 匹配逻辑
    for country in processed_input:
        if country in remaining_global:
            remaining_global.remove(country)
            popped_list.append(country)
            print(f"✅ 匹配成功: {country}")
        else:
            unmatched_list.append(country)
            print(f"❌ 未匹配: {country} (剩余可选项示例: {remaining_global[:3]}...)")

    return remaining_global, popped_list, unmatched_list


def match_data(
        find_style: str,
        dataframe: pd.DataFrame,
        column_country: str = "name",
        column_data: str = "data",
        fuzzy: bool = True,
        conf: float = 0.6
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    批量匹配地理名称并分离匹配/未匹配数据

    :param find_style: 查询类型 ('en'/'zh'/'ISO')
    :param dataframe: 原始数据DataFrame，需包含国家名称和关联数据列
    :param column_country: 国家名称列名（默认'raw_country'）
    :param column_data: 关联数据列名（默认'associated_data'）
    :param fuzzy: 是否启用模糊匹配（默认True）
    :param conf: 模糊匹配阈值（默认0.6）
    :return: (匹配数据DataFrame, 未匹配数据DataFrame)
    """
    # 验证输入数据格式
    if not {column_country, column_data}.issubset(dataframe.columns):
        missing = {column_country, column_data} - set(dataframe.columns)
        raise ValueError(f"缺少必要列: {missing}")

    # 准备结果容器
    matched_records = []
    unmatched_records = []

    # 批量处理
    for _, row in dataframe.iterrows():
        country = row[column_country]
        data = row[column_data]

        # 执行地理匹配
        matches = search_geo(
            query_type=find_style,
            query_content=str(country),
            fuzzy=fuzzy,
            cutoff=conf,
            limit=1  # 取匹配度最高的结果
        )

        # 分类存储结果
        if matches:
            matched_records.append({
                "name": matches[0],
                column_data: data
            })
        else:
            unmatched_records.append({
                column_country: country,
                column_data: data
            })

    # 构建返回DataFrame
    matched_df = pd.DataFrame(matched_records)
    unmatched_df = pd.DataFrame(unmatched_records)

    # 保留原始索引顺序
    if not matched_df.empty:
        matched_df = matched_df.set_index(dataframe.index[matched_df.index])
    if not unmatched_df.empty:
        unmatched_df = unmatched_df.set_index(dataframe.index[unmatched_df.index])

    return matched_df, unmatched_df


def plot_geo(
        dataframe: pd.DataFrame,
        gray_countries: list,
        geo_data_path: str = "./geo/WGS1984/",
        column_data: str = "value",
        cmap_colors: list = ["#72B6A1", "#95A3C3", "#E99675"],
        figuresize: tuple = (10, 5),
        dpi: int = 190
) -> None:
    """
    地理分布可视化函数

    :param dataframe: 已匹配数据框，需包含标准化国家名称和数据列
    :param gray_countries: 需要显示为灰色的国家名称列表
    :param geo_data_path: GeoJSON文件存储路径
    :param column_data: 用于染色的数据列名
    :param cmap_colors: 自定义颜色渐变列表
    :param figuresize: 图像尺寸
    :param dpi: 图像分辨率
    """
    # 创建绘图对象
    fig = plt.figure(figsize=figuresize, dpi=dpi)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # 配置颜色映射
    cmap_custom = LinearSegmentedColormap.from_list("custom_cmap", cmap_colors)
    norm = Normalize(vmin=dataframe[column_data].min(),
                     vmax=dataframe[column_data].max())

    # 专业地图要素配置
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5, alpha=0.8)

    # 智能经纬网格配置
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.3,
        color='gray',
        alpha=0.5,
        linestyle='--'
    )
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = {'size': 8, 'color': 'gray'}
    gl.ylabel_style = {'size': 8, 'color': 'gray'}

    # 增强型回归线标注
    for lat, label in [(23.436, 'Tropic of Cancer'), (-23.436, 'Tropic of Capricorn')]:
        ax.axhline(lat, color='#d62728', linewidth=0.8, linestyle=':', alpha=0.7)
        ax.text(-135, lat + 2, label,
                ha='right', va='center',
                color='#d62728',
                fontsize=7,
                fontstyle='italic')

    # 数据驱动染色逻辑
    def plot_countries(country_list, color, alpha=0.9, is_data=False):
        for country in country_list:
            try:
                gdf = gpd.read_file(f"{geo_data_path}{country}.json")
                if is_data:
                    value = dataframe.loc[dataframe['name'] == country, column_data].values[0]
                    fill_color = cmap_custom(norm(value))
                else:
                    fill_color = color

                gdf.plot(
                    ax=ax,
                    color=fill_color,
                    edgecolor='black',
                    linewidth=0.2,
                    alpha=alpha,
                    transform=ccrs.PlateCarree()
                )
            except Exception as e:
                print(f"绘制 {country} 失败: {str(e)}")

    # 分层绘制：先灰色背景，再数据染色
    plot_countries(gray_countries, '#cccccc', alpha=0.7)
    plot_countries(dataframe['name'].tolist(), None, is_data=True)

    # 专业色条配置
    sm = ScalarMappable(norm=norm, cmap=cmap_custom)
    cbar = plt.colorbar(
        sm, ax=ax,
        orientation='vertical',
        shrink=0.9,
        aspect=15,
        pad=0.03,
        label=column_data
    )
    cbar.outline.set_linewidth(0.3)
    cbar.ax.tick_params(width=0.3, labelsize=7)
    cbar.set_label(column_data, fontsize=8, labelpad=2)

    # 图框样式增强
    ax.spines['geo'].set_linewidth(0.8)
    ax.spines['geo'].set_color('#444444')

    # 自动布局优化
    plt.tight_layout(pad=1.5)
    plt.savefig('geo_visualization.png', bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == '__main__':
    # citys = find_city(target="龙港", conf=0.9)
    # print(f"所有匹配的地域名称: {citys}")
    #
    # find_file(target=citys[1])
    # print(f"匹配的文件名: {find_file(target=citys[1])}")

    print(search_geo(query_type='en',query_content='Equatorial', fuzzy=True))      # 返回'日本'
    print(search_geo('zh', '中国'))       # 返回'中国'
    print(search_geo('ISO', 'CN'))        # 返回'中国'


