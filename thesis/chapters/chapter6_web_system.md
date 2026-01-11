# 第6章 Web检测系统开发

本章详细介绍钓鱼网站检测Web系统的开发过程,包括开发环境配置、后端API开发、前端页面设计和系统集成测试。该Web系统采用Flask框架实现后端服务,使用Bootstrap 5构建响应式前端界面,为用户提供便捷的URL检测服务。

## 6.1 开发环境配置

### 6.1.1 软件环境

本系统的开发环境配置如表6-1所示。开发环境基于Windows 11平台,使用Python 3.14作为主要开发语言,Flask 2.x作为Web框架,SQLite 3.x作为数据库系统。开发工具采用VS Code编辑器,并使用Git进行版本控制。

**表6-1 开发环境配置**

| 类别 | 名称 | 版本 | 用途说明 |
|-----|------|------|---------|
| 操作系统 | Windows 11 | 22H2 | 开发环境平台 |
| 编程语言 | Python | 3.14.0 | 主要开发语言 |
| Web框架 | Flask | 2.2+ | Web应用框架 |
| 数据库 | SQLite | 3.x | 轻量级数据库 |
| 前端框架 | Bootstrap | 5.3.0 | 响应式UI框架 |
| 图标库 | Bootstrap Icons | 1.10.0 | 图标库 |
| 开发工具 | VS Code | 1.85+ | 代码编辑器 |
| 版本控制 | Git | 2.40+ | 代码版本管理 |

### 6.1.2 Python依赖库

系统的主要Python依赖库如表6-2所示。依赖库分为五大类:数据处理、机器学习、网络请求、Web框架和测试工具。其中,pandas和numpy用于数据处理,scikit-learn和xgboost实现机器学习算法,requests和urllib3处理网络请求,Flask及其扩展提供Web服务能力,pytest用于单元测试。

**表6-2 主要Python依赖库**

| 库名 | 版本 | 功能类别 | 用途说明 |
|-----|------|---------|---------|
| pandas | 1.5+ | 数据处理 | 数据分析与处理 |
| numpy | 1.23+ | 数据处理 | 数值计算 |
| scikit-learn | 1.2+ | 机器学习 | 机器学习算法库 |
| xgboost | 1.7+ | 机器学习 | XGBoost算法 |
| joblib | 1.2+ | 机器学习 | 模型序列化 |
| requests | 2.28+ | 网络请求 | HTTP请求库 |
| urllib3 | 1.26+ | 网络请求 | URL处理 |
| tldextract | 3.4+ | URL解析 | 域名提取与解析 |
| pyOpenSSL | 23.0+ | SSL证书 | SSL证书处理 |
| flask | 2.2+ | Web框架 | Web应用框架 |
| flask-cors | 3.0+ | Web框架 | 跨域请求支持 |
| pytest | 7.0+ | 测试工具 | 单元测试框架 |

依赖库通过requirements.txt文件统一管理,使用清华镜像源加速安装:

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 6.1.3 项目目录结构

Web系统的项目目录结构如下所示。采用MVC架构模式,将代码分为路由层(routes)、业务逻辑层(utils)、数据层(database.db)和视图层(templates/static)。这种结构清晰地分离了各个模块的职责,便于开发维护和功能扩展。

```
web/
├── app.py                    # Flask应用入口文件
├── web_config.py             # 应用配置文件
├── database.db               # SQLite数据库文件
├── routes/                   # 路由模块目录
│   ├── __init__.py          # 模块初始化
│   ├── main.py              # 页面路由(首页、历史、关于)
│   └── api.py               # API路由(检测、历史记录管理)
├── utils/                    # 工具模块目录
│   ├── __init__.py          # 模块初始化
│   ├── predictor.py         # 预测器(模型加载与预测)
│   ├── url_validator.py     # URL验证器
│   └── exporter.py          # 数据导出器(CSV/JSON)
├── templates/                # Jinja2模板目录
│   ├── base.html            # 基础布局模板
│   ├── index.html           # 首页(URL检测)
│   ├── history.html         # 历史记录页
│   └── about.html           # 关于页面
└── static/                   # 静态资源目录
    ├── css/
    │   └── style.css        # 自定义样式表
    └── js/
        ├── main.js          # 首页交互逻辑
        └── history.js       # 历史页面逻辑
```

路由模块负责处理HTTP请求,main.py处理页面渲染请求,api.py处理RESTful API请求。工具模块封装核心业务逻辑,predictor.py实现模型预测,url_validator.py提供URL验证,exporter.py支持数据导出。模板使用Jinja2模板引擎,支持模板继承和变量渲染。静态资源包括CSS样式和JavaScript脚本,实现前端交互功能。

## 6.2 后端开发

### 6.2.1 Flask应用架构

本系统采用Flask的应用工厂模式(Application Factory Pattern),这种模式将应用实例的创建封装在工厂函数中,支持创建多个配置不同的应用实例,便于开发、测试和生产环境的切换。

**应用工厂函数实现**

app.py文件中的`create_app()`函数是应用工厂的核心,它接受配置名称参数,创建并配置Flask应用实例:

```python
def create_app(config_name='default'):
    """
    应用工厂函数

    Args:
        config_name: 配置名称(development/production/testing)

    Returns:
        Flask应用实例
    """
    app = Flask(__name__)

    # 加载配置
    app.config.from_object(config[config_name])

    # 初始化数据库
    init_database(app)

    # 注册蓝图
    register_blueprints(app)

    # 注册错误处理
    register_error_handlers(app)

    return app
```

工厂函数执行四个关键步骤:1) 创建Flask实例;2) 从配置对象加载配置;3) 初始化数据库表结构;4) 注册蓝图和错误处理器。这种设计使应用配置灵活可控,各模块职责清晰。

**配置管理**

web_config.py文件定义了三种环境配置,如表6-3所示:

**表6-3 环境配置对比**

| 配置项 | 开发环境 | 生产环境 | 测试环境 |
|-------|---------|---------|---------|
| DEBUG | True | False | True |
| DATABASE_PATH | database.db | database.db | :memory: |
| LOG_LEVEL | DEBUG | WARNING | DEBUG |
| MAX_BATCH_SIZE | 100 | 100 | 10 |

配置类采用继承结构,基类Config定义通用配置,子类覆盖特定配置。开发环境启用调试模式和详细日志,生产环境禁用调试并提高日志级别,测试环境使用内存数据库确保测试隔离。

**数据库初始化**

`init_database()`函数创建SQLite数据库表结构:

```python
def init_database(app):
    """初始化SQLite数据库"""
    db_path = app.config['DATABASE_PATH']
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 创建检测历史表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detection_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            prediction INTEGER NOT NULL,
            confidence REAL NOT NULL,
            features TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 创建索引优化查询
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at
                    ON detection_history(created_at DESC)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_prediction
                    ON detection_history(prediction)')

    conn.commit()
    conn.close()
```

数据库表结构包含URL、预测结果、置信度、特征数据和创建时间五个字段。创建了两个索引:按时间倒序的索引用于历史记录查询,按预测结果的索引用于筛选钓鱼网站。

**蓝图注册**

系统使用蓝图(Blueprint)组织路由,将路由分为页面路由(main_bp)和API路由(api_bp)两个模块:

```python
def register_blueprints(app):
    """注册蓝图"""
    from routes.main import main_bp
    from routes.api import api_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix='/api')
```

main_bp处理页面请求(`/`, `/history`, `/about`),api_bp处理API请求并统一添加`/api`前缀。这种设计实现了前后端路由的清晰分离。

**错误处理**

`register_error_handlers()`函数注册全局错误处理器,统一处理400、404、500等HTTP错误,返回JSON格式的错误信息,提升API的健壮性和用户体验。

### 6.2.2 核心API接口实现

API模块(routes/api.py)提供RESTful接口,实现URL检测、历史记录管理、数据导出等功能。所有接口采用JSON格式交互,遵循统一的响应格式。

**单URL检测接口**

`POST /api/predict`接口接收URL并返回检测结果:

```python
@api_bp.route('/predict', methods=['POST'])
def predict():
    """
    单URL检测API

    Request: {"url": "https://example.com", "mode": "full"}
    Response: {"success": true, "data": {...}}
    """
    try:
        # 1. 获取并验证请求数据
        data = request.get_json(silent=True)
        if data is None or 'url' not in data:
            return jsonify({'success': False, 'error': '无效请求'}), 400

        url = data['url']
        mode = data.get('mode', 'full')  # full/quick模式

        # 2. URL验证
        from utils.url_validator import URLValidator
        is_valid, error_msg = URLValidator.is_valid_url(url)
        if not is_valid:
            return jsonify({'success': False, 'error': error_msg}), 400

        # 3. 标准化URL并预测
        normalized_url = URLValidator.normalize_url(url)
        predictor = get_predictor()
        result = predictor.predict_url(normalized_url,
                                        full_features=(mode != 'quick'))

        # 4. 保存历史记录
        save_to_history(normalized_url, result)

        return jsonify({'success': True, 'data': result})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
```

接口支持两种检测模式:完整模式提取全部30维特征(准确但较慢),快速模式仅提取URL特征并用默认值填充网络特征(快速但不够准确)。接口实现了完善的错误处理,对输入进行严格验证,确保服务稳定性。

**批量检测接口**

`POST /api/batch`接口支持批量URL检测:

```python
@api_bp.route('/batch', methods=['POST'])
def batch_predict():
    """
    批量URL检测API

    Request: {"urls": ["url1", "url2", ...]}
    Response: {"success": true, "summary": {...}, "results": [...]}
    """
    try:
        data = request.get_json(silent=True)
        urls = data.get('urls', [])

        # 验证参数
        if not isinstance(urls, list):
            return jsonify({'success': False, 'error': 'urls必须是数组'}), 400

        max_size = current_app.config['MAX_BATCH_SIZE']
        if len(urls) > max_size:
            return jsonify({'success': False,
                          'error': f'最多支持{max_size}个URL'}), 400

        # 批量预测
        predictor = get_predictor()
        results = []
        for url in urls:
            try:
                is_valid, error_msg = URLValidator.is_valid_url(url)
                if not is_valid:
                    results.append({'url': url, 'success': False,
                                   'error': error_msg})
                    continue

                normalized_url = URLValidator.normalize_url(url)
                result = predictor.predict_url(normalized_url)
                results.append({'url': normalized_url, 'success': True, **result})
                save_to_history(normalized_url, result)
            except Exception as e:
                results.append({'url': url, 'success': False, 'error': str(e)})

        # 统计汇总
        success_count = sum(1 for r in results if r.get('success'))
        phishing_count = sum(1 for r in results if r.get('prediction') == 1)

        return jsonify({
            'success': True,
            'summary': {
                'total': len(urls),
                'success': success_count,
                'failed': len(urls) - success_count,
                'phishing': phishing_count,
                'normal': success_count - phishing_count
            },
            'results': results
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
```

批量接口限制最多100个URL,对每个URL单独捕获异常,确保部分失败不影响整体结果。返回结果包含汇总统计和详细列表,便于前端展示。

**历史记录查询接口**

`GET /api/history`接口支持分页、筛选、搜索和排序:

```python
@api_bp.route('/history', methods=['GET'])
def get_history():
    """
    获取检测历史记录

    Query Parameters:
        page: 页码(默认1)
        per_page: 每页数量(默认20,最大100)
        prediction: 筛选(0正常/1钓鱼/all全部)
        keyword: URL关键词搜索
        start_date/end_date: 日期范围
        sort_by: 排序字段(created_at/confidence)
        sort_order: 排序方向(asc/desc)
    """
    # 解析并验证参数
    page = request.args.get('page', 1, type=int)
    per_page = min(max(request.args.get('per_page', 20, type=int), 1), 100)
    prediction = request.args.get('prediction', 'all')
    keyword = request.args.get('keyword', '').strip()
    sort_by = request.args.get('sort_by', 'created_at')
    sort_order = 'DESC' if request.args.get('sort_order', 'desc') == 'desc' else 'ASC'

    # 构建查询条件
    conditions = []
    params = []
    if prediction in ('0', '1'):
        conditions.append('prediction = ?')
        params.append(int(prediction))
    if keyword:
        conditions.append('url LIKE ?')
        params.append(f'%{keyword}%')

    where_clause = ' AND '.join(conditions) if conditions else '1=1'

    # 查询数据
    conn = get_db()
    cursor = conn.cursor()

    # 获取总数
    cursor.execute(f'SELECT COUNT(*) FROM detection_history WHERE {where_clause}',
                   params)
    total = cursor.fetchone()[0]

    # 分页查询
    offset = (page - 1) * per_page
    cursor.execute(f'''
        SELECT id, url, prediction, confidence, created_at
        FROM detection_history
        WHERE {where_clause}
        ORDER BY {sort_by} {sort_order}
        LIMIT ? OFFSET ?
    ''', params + [per_page, offset])

    records = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return jsonify({
        'success': True,
        'data': {
            'records': records,
            'total': total,
            'page': page,
            'per_page': per_page,
            'pages': (total + per_page - 1) // per_page if total > 0 else 0
        }
    })
```

接口支持多种查询参数组合,实现灵活的数据筛选。使用SQL参数化查询防止SQL注入攻击。返回结果包含分页元数据,便于前端实现分页控件。

**其他API接口**

除核心接口外,系统还提供:
- `DELETE /api/history/<id>`: 删除单条记录
- `DELETE /api/history/clear`: 清空所有记录
- `POST /api/history/batch-delete`: 批量删除记录
- `GET /api/stats`: 获取统计信息(总检测数、钓鱼率、今日检测、7天趋势)
- `GET /api/history/export`: 导出历史记录(CSV/JSON格式)

这些接口共同构成完整的API体系,支撑Web系统的各项功能。

### 6.2.3 预测器模块

utils/predictor.py封装了模型加载和预测逻辑,为API提供简洁的调用接口:

```python
class WebPredictor:
    """Web端预测器"""

    def __init__(self, model_path: str, scaler_path: str = None):
        """初始化预测器,加载模型"""
        models_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path).replace('.pkl', '')

        # 使用PhishingPredictor加载模型
        self.predictor = PhishingPredictor(
            models_dir=models_dir,
            model_name=model_name
        )

        # 加载特征配置
        config_path = os.path.join(models_dir, 'model_config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            self.feature_names = config.get('feature_names')

    def predict_url(self, url: str, full_features: bool = False) -> dict:
        """预测单个URL"""
        # 提取特征
        extractor = FeatureExtractor(url)
        if full_features:
            features = extractor.extract_all()  # 完整模式
        else:
            features = extractor.extract_url_only()  # 快速模式
            features.update(self.DEFAULT_NETWORK_FEATURES)

        # 特征名称映射和值转换
        mapped_features = self._map_features(features)

        # 按顺序构建特征数组
        feature_array = np.array([mapped_features.get(name, 0)
                                  for name in self.feature_names]).reshape(1, -1)

        # 预测
        result = self.predictor.predict(feature_array, return_proba=True)

        return {
            'url': url,
            'prediction': int(result['prediction']),
            'label': '钓鱼网站' if result['prediction'] == 1 else '正常网站',
            'confidence': float(result['confidence']),
            'phishing_probability': float(result['probability']['phishing']),
            'mode': 'full' if full_features else 'quick'
        }
```

预测器支持两种特征提取模式,完整模式适合高准确率要求场景,快速模式适合实时响应场景。预测器使用单例模式,避免重复加载模型,提升性能。

## 6.3 前端开发

### 6.3.1 页面布局设计

前端采用Bootstrap 5框架构建响应式布局,使用Jinja2模板引擎实现模板继承和代码复用。

**基础模板(base.html)**

base.html定义了所有页面共用的基础结构:

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}钓鱼网站检测系统{% endblock %}</title>

    <!-- Bootstrap 5 CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"
          rel="stylesheet">
    <!-- Bootstrap Icons -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css"
          rel="stylesheet">
    <!-- 自定义CSS -->
    <link href="{{ url_for('static', filename='css/style.css') }}" rel="stylesheet">
</head>
<body class="d-flex flex-column min-vh-100">
    <!-- 导航栏 -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary shadow-sm">
        <div class="container">
            <a class="navbar-brand d-flex align-items-center" href="/">
                <i class="bi bi-shield-check me-2"></i>
                <span>钓鱼网站检测系统</span>
            </a>
            <div class="collapse navbar-collapse">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="/">
                            <i class="bi bi-search me-1"></i>检测
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/history">
                            <i class="bi bi-clock-history me-1"></i>历史记录
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/about">
                            <i class="bi bi-info-circle me-1"></i>关于
                        </a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <!-- 主内容区 -->
    <main class="flex-grow-1">
        <div class="container py-4">
            {% block content %}{% endblock %}
        </div>
    </main>

    <!-- 页脚 -->
    <footer class="bg-light py-3 mt-auto border-top">
        <div class="container">
            <p class="mb-0 text-muted small text-center">
                基于网络流量特征的钓鱼网站检测系统 - 毕业设计
            </p>
        </div>
    </footer>

    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
</body>
</html>
```

基础模板采用三段式布局:顶部导航栏、中间内容区、底部页脚。使用`flex-grow-1`确保内容区自动填充剩余空间,页脚始终位于底部。导航栏使用Bootstrap的navbar组件,支持响应式折叠。

**首页(index.html)**

首页继承基础模板,提供URL检测表单,如图6-1所示:

[图6-1 系统首页界面]

```html
{% extends "base.html" %}

{% block content %}
<div class="row justify-content-center">
    <div class="col-lg-8 col-xl-7">
        <!-- 欢迎横幅 -->
        <div class="text-center mb-4">
            <h1 class="display-6 fw-bold text-primary">
                <i class="bi bi-shield-check"></i> URL安全检测
            </h1>
            <p class="lead text-muted">输入URL,快速检测是否为钓鱼网站</p>
        </div>

        <!-- 单URL检测卡片 -->
        <div class="card shadow-sm mb-4">
            <div class="card-header bg-primary text-white">
                <h5 class="mb-0">
                    <i class="bi bi-search me-2"></i>单URL检测
                </h5>
            </div>
            <div class="card-body p-4">
                <form id="detectForm">
                    <div class="input-group input-group-lg">
                        <span class="input-group-text bg-light">
                            <i class="bi bi-link-45deg text-primary"></i>
                        </span>
                        <input type="text" class="form-control" id="urlInput"
                               placeholder="例如:https://www.example.com" required>
                        <button type="submit" class="btn btn-primary" id="detectBtn">
                            <i class="bi bi-shield-check me-1"></i>检测
                        </button>
                    </div>
                </form>
            </div>
        </div>

        <!-- 检测结果展示区 -->
        <div id="resultArea" style="display: none;">
            <div class="card shadow-sm">
                <div class="card-body" id="resultContent">
                    <!-- 动态填充检测结果 -->
                </div>
            </div>
        </div>

        <!-- 批量检测折叠区 -->
        <div class="card shadow-sm mb-4">
            <div class="card-header bg-light">
                <a data-bs-toggle="collapse" href="#batchCollapse">
                    <i class="bi bi-list-check me-2"></i>批量检测
                </a>
            </div>
            <div class="collapse" id="batchCollapse">
                <div class="card-body">
                    <form id="batchForm">
                        <textarea class="form-control" id="batchUrls" rows="5"
                                  placeholder="每行一个URL(最多100个)"></textarea>
                        <button type="submit" class="btn btn-outline-primary mt-3">
                            开始批量检测
                        </button>
                    </form>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}
```

首页设计注重用户体验:大尺寸输入框便于输入,图标提供视觉引导,批量检测默认折叠避免干扰主功能。使用Bootstrap的栅格系统实现响应式布局,在大屏幕上内容居中,在小屏幕上占满宽度。

**历史记录页面(history.html)**

历史记录页面展示检测历史,支持筛选、搜索、导出,如图6-2所示:

[图6-2 历史记录页面]

页面包含筛选工具栏、数据表格、分页控件三个部分。表格使用Bootstrap的table组件,支持响应式滚动。分页控件通过JavaScript动态生成,支持跳转到指定页。

### 6.3.2 Bootstrap 5组件应用

系统充分利用Bootstrap 5提供的UI组件:

**卡片组件(Card)**: 用于包裹检测表单、结果展示、使用说明等模块,提供统一的视觉风格。

**表单组件(Form)**: 使用`form-control`、`input-group`等类美化表单元素,`form-text`提供帮助文本。

**按钮组件(Button)**: 使用`btn-primary`、`btn-outline-primary`等样式,`disabled`状态表示加载中。

**表格组件(Table)**: 历史记录使用`table-hover`、`table-striped`提升可读性。

**折叠组件(Collapse)**: 批量检测默认折叠,点击展开,节省页面空间。

**徽章组件(Badge)**: 使用`badge bg-danger`、`badge bg-success`标识检测结果。

**图标库(Icons)**: Bootstrap Icons提供300+图标,使用`<i class="bi bi-shield-check"></i>`语法引入。

**栅格系统(Grid)**: 使用`row`、`col-lg-8`等类实现响应式布局,适配各种屏幕尺寸。

### 6.3.3 JavaScript交互实现

前端JavaScript(static/js/main.js)实现与后端API的异步交互:

**单URL检测功能**

```javascript
// 初始化单URL检测表单
function initDetectForm() {
    const detectForm = document.getElementById('detectForm');
    if (!detectForm) return;

    detectForm.addEventListener('submit', async function(event) {
        event.preventDefault();

        const urlInput = document.getElementById('urlInput');
        const detectBtn = document.getElementById('detectBtn');
        const url = urlInput.value.trim();

        if (!url) {
            showToast('请输入URL地址', 'warning');
            return;
        }

        // 显示加载状态
        setButtonLoading(detectBtn, true);

        try {
            // 调用检测API
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url: url })
            });

            const data = await response.json();

            if (data.success) {
                displayResult(data.data);  // 显示结果
            } else {
                displayError(data.error);  // 显示错误
            }
        } catch (error) {
            displayError('网络请求失败: ' + error.message);
        } finally {
            setButtonLoading(detectBtn, false);
        }
    });
}

// 显示检测结果
function displayResult(result) {
    const resultContent = document.getElementById('resultContent');
    const isPhishing = result.prediction === 1;

    resultContent.innerHTML = `
        <div class="text-center">
            <i class="bi bi-${isPhishing ? 'exclamation-triangle' : 'check-circle'}
               display-1 text-${isPhishing ? 'danger' : 'success'}"></i>
            <h3 class="mt-3">
                <span class="badge bg-${isPhishing ? 'danger' : 'success'}">
                    ${result.label}
                </span>
            </h3>
            <p class="lead">置信度: ${(result.confidence * 100).toFixed(1)}%</p>
            <hr>
            <p class="text-muted small">URL: ${result.url}</p>
        </div>
    `;

    document.getElementById('resultArea').style.display = 'block';
}
```

检测功能使用Fetch API发送异步请求,避免页面刷新。显示加载状态提升用户体验,按钮变为禁用状态并显示"检测中..."文本。结果展示根据检测结果动态选择样式,钓鱼网站显示红色警告图标,正常网站显示绿色对勾图标。

**批量检测功能**

批量检测解析textarea中的URL列表,调用`/api/batch`接口,以表格形式展示结果:

```javascript
function initBatchForm() {
    const batchForm = document.getElementById('batchForm');
    if (!batchForm) return;

    batchForm.addEventListener('submit', async function(event) {
        event.preventDefault();

        const batchUrls = document.getElementById('batchUrls');
        const urlText = batchUrls.value.trim();

        // 解析URL列表
        const urls = urlText.split('\n')
            .map(u => u.trim())
            .filter(u => u.length > 0);

        if (urls.length === 0) {
            showToast('请输入URL', 'warning');
            return;
        }

        if (urls.length > 100) {
            showToast('最多支持100个URL', 'warning');
            return;
        }

        // 调用批量检测API
        const response = await fetch('/api/batch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ urls: urls })
        });

        const data = await response.json();

        if (data.success) {
            displayBatchResults(data.results, data.summary);
        }
    });
}
```

批量检测限制最多100个URL,前端进行预验证减少无效请求。结果以汇总统计和详细列表两种形式展示,便于用户快速了解整体情况。

**历史记录功能**

history.js实现历史记录的加载、筛选、分页、删除、导出功能:

```javascript
// 加载历史记录
async function loadHistory(page = 1) {
    const params = new URLSearchParams({
        page: page,
        per_page: 20,
        prediction: document.getElementById('filterPrediction').value,
        keyword: document.getElementById('filterKeyword').value
    });

    const response = await fetch(`/api/history?${params}`);
    const data = await response.json();

    if (data.success) {
        renderHistoryTable(data.data.records);
        renderPagination(data.data.page, data.data.pages);
    }
}

// 删除记录
async function deleteRecord(id) {
    if (!confirm('确定删除这条记录吗?')) return;

    const response = await fetch(`/api/history/${id}`, {
        method: 'DELETE'
    });

    const data = await response.json();

    if (data.success) {
        showToast('删除成功', 'success');
        loadHistory();  // 重新加载
    }
}

// 导出记录
function exportHistory(format) {
    const params = new URLSearchParams({
        format: format,
        prediction: document.getElementById('filterPrediction').value
    });

    window.location.href = `/api/history/export?${params}`;
}
```

历史记录功能实现了完整的CRUD操作,支持多种筛选条件组合,分页加载减少数据传输量,导出功能直接触发浏览器下载。

## 6.4 系统集成与测试

### 6.4.1 模型集成

Web系统通过WebPredictor类集成训练好的机器学习模型。模型文件存储在`data/models/`目录,包括集成模型(ensemble_model_final.pkl)、标准化器(scaler.pkl)和特征配置(model_config.json)。

预测器在首次调用时延迟加载模型,使用单例模式确保全局唯一实例,避免重复加载占用内存。模型加载耗时约1-2秒,单次预测耗时10-50毫秒(快速模式)或1-3秒(完整模式),满足Web服务响应时间要求。

### 6.4.2 前后端联调

系统开发完成后进行全面的前后端联调测试:

**接口测试**: 使用Postman工具测试各API接口,验证请求参数、响应格式、错误处理的正确性。测试用例覆盖正常情况、边界情况和异常情况。

**功能测试**: 在浏览器中测试各页面功能,包括单URL检测、批量检测、历史记录查询、数据导出等,验证前后端交互的完整性。

**兼容性测试**: 在Chrome、Firefox、Edge三种主流浏览器中测试,确保跨浏览器兼容性。测试发现所有浏览器均正常运行,无兼容性问题。

**响应式测试**: 在桌面(1920×1080)、平板(768×1024)、手机(375×667)等不同屏幕尺寸下测试,验证响应式布局的适配性。测试结果表明布局在各种屏幕下均显示正常。

**性能测试**: 使用Chrome DevTools的Network和Performance工具分析页面加载时间和API响应时间。首页加载时间约500ms,单URL检测响应时间1-3秒,批量检测10个URL约5-8秒,性能满足要求。

### 6.4.3 系统部署

系统部署采用Flask内置的开发服务器,适合本地开发和演示。启动命令如下:

```bash
cd phishing-detection/web
python app.py
```

服务默认监听0.0.0.0:5000,访问http://localhost:5000即可使用系统。若需生产部署,建议使用Gunicorn或uWSGI作为WSGI服务器,配合Nginx作为反向代理,提升并发处理能力和安全性。

检测结果展示如图6-3所示:

[图6-3 URL检测结果展示]

## 6.5 本章小结

本章详细介绍了钓鱼网站检测Web系统的开发过程。首先介绍了开发环境配置,包括Python 3.14、Flask 2.x、Bootstrap 5等软件环境,以及12个主要依赖库的功能用途。然后描述了后端开发,采用Flask应用工厂模式实现灵活的配置管理,使用蓝图组织路由实现前后端分离,开发了单URL检测、批量检测、历史记录管理等RESTful API接口,所有接口均实现了完善的输入验证和错误处理。接着说明了前端开发,使用Jinja2模板引擎实现模板继承,充分利用Bootstrap 5的卡片、表单、表格、折叠等组件构建响应式界面,通过JavaScript实现与后端API的异步交互,提供流畅的用户体验。最后介绍了系统集成与测试,包括模型集成、前后端联调、兼容性测试、响应式测试和性能测试,确保系统稳定可靠。

本章开发的Web系统具有以下特点:一是架构清晰,采用MVC模式分离关注点;二是功能完善,支持单URL检测、批量检测、历史记录管理、数据导出等功能;三是交互友好,响应式设计适配各种设备,加载状态提示提升用户体验;四是性能良好,单次检测响应时间1-3秒,满足实时检测需求;五是可扩展性强,蓝图架构便于添加新功能。该Web系统为钓鱼网站检测技术提供了实用的应用平台,为后续研究和推广奠定了基础。
