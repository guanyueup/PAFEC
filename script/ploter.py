import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import List, Sequence, Optional, Tuple, Union

class IEEEPlotter:
    """
    IEEE 风格画图工具
    - 默认单栏宽 3.5 in（≈8.9 cm），Times 系 8pt，y 向虚线网格，向量输出友好
    - 自动配色（Okabe–Ito 色盲友好）与纹理（hatch），黑白打印可读
    - color_mode:
        'auto'           -> 彩色 + 纹理（柱状图），折线/散点用颜色+标记/线型
        'color-only'     -> 仅彩色，不用纹理（柱状图）；线图/散点仍用标记/线型区分
        'texture-only'   -> 柱状图白底 + 纹理；线图/散点用黑白灰 + 标记/线型
        'bw'             -> 与 'texture-only' 同义（黑白打印友好）
    """

    OKABE_ITO = ['#0072B2', '#E69F00', '#009E73', '#D55E00',
                 '#CC79A7', '#F0E442', '#56B4E9', '#000000']
    GRAYS = ['#000000', '#404040', '#7F7F7F', '#B0B0B0', '#D0D0D0']
    HATCHES = ['//', '\\\\', 'xx', '++', '..', '||', '--', 'oo']  # 常用纹理组合
    MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
    LINESTYLES = ['-', '--', '-.', ':']

    COL_WIDTHS_IN = {'single': 3.5, 'onehalf': 5.0, 'double': 7.16}

    def __init__(self,
                 base_size: int = 8,
                 font_family: Tuple[str, str] = ('Times New Roman', 'DejaVu Serif'),
                 column: str = 'single',
                 color_mode: str = 'auto'):
        self.base_size = base_size
        self.font_family = font_family
        self.column = column if column in self.COL_WIDTHS_IN else 'single'
        self.color_mode = color_mode
        self._apply_rc()

    def _apply_rc(self):
        mpl.rcParams.update({
            'font.size': self.base_size,
            'font.family': 'serif',
            'font.serif': list(self.font_family),
            'axes.titlesize': self.base_size,
            'axes.labelsize': self.base_size,
            'xtick.labelsize': max(self.base_size - 1, 6),
            'ytick.labelsize': max(self.base_size - 1, 6),
            'legend.fontsize': max(self.base_size - 1, 6),
            'axes.linewidth': 0.8,
            'grid.linestyle': '--',
            'grid.linewidth': 0.5,
            'grid.alpha': 0.5,
            'axes.grid': True,
            'axes.grid.axis': 'y',
            'savefig.dpi': 300,
            'figure.dpi': 150,
            'hatch.linewidth': 0.6,
            # 'hatch.color': 'black',  # Matplotlib>=3.7 可启用
            'pdf.fonttype': 42,       # 嵌入 TrueType，便于矢量编辑
            'ps.fonttype': 42,
            'axes.unicode_minus': False,
        })

    # ---------- 基础：新建画布 ----------
    def figure(self, height: float = 2.2, width: Optional[float] = None) -> Tuple[plt.Figure, plt.Axes]:
        if width is None:
            width = self.COL_WIDTHS_IN.get(self.column, 3.5)
        fig, ax = plt.subplots(figsize=(width, height))
        # 微调边框/层级
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_axisbelow(True)
        ax.margins(x=0.02)
        return fig, ax

    # ---------- 自动样式选择 ----------
    def _get_color(self, i: int) -> str:
        if self.color_mode in ('texture-only', 'bw'):
            return self.GRAYS[i % len(self.GRAYS)]
        return self.OKABE_ITO[i % len(self.OKABE_ITO)]

    def _get_hatch(self, i: int) -> str:
        return self.HATCHES[i % len(self.HATCHES)]

    def _get_marker(self, i: int) -> str:
        return self.MARKERS[i % len(self.MARKERS)]

    def _get_linestyle(self, i: int) -> str:
        return self.LINESTYLES[i % len(self.LINESTYLES)]

    def bar(self,
        categories: Sequence[Union[str, int]],
        series_list: Union[Sequence[float], Sequence[Sequence[float]]],
        labels: Optional[Sequence[str]] = None,
        yerr: Optional[Union[Sequence[float], Sequence[Sequence[float]]]] = None,
        show_values: bool = False,
        value_fmt: str = '%.0f',
        total_width: float = 0.8,
        edgecolor: str = 'black',
        linewidth: float = 0.8,
        colors: Optional[Sequence[str]] = None,
        hatches: Optional[Sequence[str]] = None,
        height: float = 2.2,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        legend_loc: str = 'best',
        ylim: Optional[Tuple[float, float]] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        添加了 ylim 参数，允许用户设置 y 轴的范围。
        """

        # 统一 series 形状
        if isinstance(series_list[0], (int, float, np.floating)):
            series_list = [series_list]  # 单系列转列表
        series_list = [np.asarray(s, dtype=float) for s in series_list]
        n_series = len(series_list)
        n_cats = len(categories)
        assert all(len(s) == n_cats for s in series_list), "每个系列长度需等于类别数"

        # 自动标签
        if labels is None:
            labels = [f'S{i+1}' for i in range(n_series)]

        # 自动样式
        if colors is None:
            colors = [self._get_color(i) for i in range(n_series)]
        if hatches is None and self.color_mode in ('color-only',):
            hatches = [None] * n_series
        elif hatches is None:
            hatches = [self._get_hatch(i) for i in range(n_series)]

        # 坐标与画布
        fig, ax = self.figure(height=height)
        x = np.arange(n_cats)
        bar_width = total_width / max(n_series, 1)
        offsets = (np.arange(n_series) - (n_series - 1) / 2.0) * bar_width

        # 处理误差
        if yerr is not None:
            if isinstance(yerr[0], (int, float, np.floating)):
                yerr = [yerr] * n_series

        containers = []
        for i, (vals, lab) in enumerate(zip(series_list, labels)):
            err_i = None if yerr is None else yerr[i]
            facecolor = colors[i] if self.color_mode != 'texture-only' and self.color_mode != 'bw' else 'white'
            hatch_i = hatches[i]
            bars = ax.bar(x + offsets[i], vals,
                        width=bar_width,
                        label=lab,
                        color=facecolor,
                        edgecolor=edgecolor,
                        linewidth=linewidth,
                        hatch=hatch_i,
                        yerr=err_i,
                        capsize=2 if err_i is not None else 0,
                        zorder=3)
            containers.append(bars)
            if show_values:
                ax.bar_label(bars, fmt=value_fmt, padding=2, fontsize=mpl.rcParams['xtick.labelsize'],label_type='edge')

        # 轴与文本
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)
        if title: ax.set_title(title, pad=4)
        if labels:
            ax.legend(frameon=False, handlelength=1.6, handletextpad=0.4, loc=legend_loc)

        # 设置 y 轴范围
        if ylim is not None:
            ax.set_ylim(ylim)

        plt.tight_layout()
        return fig, ax

    # ---------- 折线图（自动颜色/线型/标记） ----------
    def line(self,
             x: Sequence[float],
             y_list: Union[Sequence[float], Sequence[Sequence[float]]],
             labels: Optional[Sequence[str]] = None,
             colors: Optional[Sequence[str]] = None,
             linestyles: Optional[Sequence[str]] = None,
             markers: Optional[Sequence[str]] = None,
             linewidth: float = 1.4,
             markersize: float = 3.5,
             height: float = 2.2,
             title: Optional[str] = None,
             xlabel: Optional[str] = None,
             ylabel: Optional[str] = None,
             legend_loc: str = 'best') -> Tuple[plt.Figure, plt.Axes]:

        if isinstance(y_list[0], (int, float, np.floating)):
            y_list = [y_list]
        y_list = [np.asarray(y, dtype=float) for y in y_list]
        n_series = len(y_list)
        x = np.asarray(x, dtype=float)

        if labels is None:
            labels = [f'S{i+1}' for i in range(n_series)]
        if colors is None:
            colors = [self._get_color(i) for i in range(n_series)]
        if linestyles is None:
            linestyles = [self._get_linestyle(i) for i in range(n_series)]
        if markers is None:
            markers = [self._get_marker(i) for i in range(n_series)]

        fig, ax = self.figure(height=height)
        for i, (y, lab) in enumerate(zip(y_list, labels)):
            color_i = colors[i] if self.color_mode != 'texture-only' and self.color_mode != 'bw' else self.GRAYS[i % len(self.GRAYS)]
            ax.plot(x, y,
                    label=lab,
                    color=color_i,
                    linestyle=linestyles[i],
                    marker=markers[i],
                    markersize=markersize,
                    linewidth=linewidth,
                    zorder=3)

        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)
        if title: ax.set_title(title, pad=4)
        if labels:
            ax.legend(frameon=False, loc=legend_loc)
        plt.tight_layout()
        return fig, ax

    # ---------- 散点图（可选拟合线） ----------
    def scatter(self,
                x: Sequence[float],
                y: Sequence[float],
                label: Optional[str] = None,
                color: Optional[str] = None,
                marker: Optional[str] = None,
                s: float = 20,
                alpha: float = 0.9,
                fit: Optional[Union[str, int]] = None,  # 'linear' 或 多项式阶数（int）
                fit_color: Optional[str] = None,
                fit_linestyle: str = '--',
                fit_linewidth: float = 1.2,
                height: float = 2.0,
                title: Optional[str] = None,
                xlabel: Optional[str] = None,
                ylabel: Optional[str] = None,
                legend_loc: str = 'best') -> Tuple[plt.Figure, plt.Axes]:

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        fig, ax = self.figure(height=height)

        c = color or (self._get_color(0) if self.color_mode not in ('texture-only', 'bw') else self.GRAYS[0])
        m = marker or self._get_marker(0)

        sc = ax.scatter(x, y, s=s, label=label, color=c, marker=m, alpha=alpha, zorder=3)

        # 拟合线
        if fit is not None:
            deg = 1 if fit == 'linear' else (int(fit) if isinstance(fit, int) else 1)
            # 仅在有效数据时拟合
            if len(x) >= deg + 1 and np.isfinite(x).all() and np.isfinite(y).all():
                coeffs = np.polyfit(x, y, deg)
                poly = np.poly1d(coeffs)
                xs = np.linspace(np.min(x), np.max(x), 200)
                ys = poly(xs)
                fc = fit_color or c
                ax.plot(xs, ys, color=fc, linestyle=fit_linestyle, linewidth=fit_linewidth,
                        label=(f'Fit (deg={deg})' if label is None else f'{label} Fit'), zorder=3)

        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)
        if title: ax.set_title(title, pad=4)
        if label or fit is not None:
            ax.legend(frameon=False, loc=legend_loc)
        plt.tight_layout()
        return fig, ax

    # ---------- CDF（经验分布函数，阶梯线） ----------
    def cdf(self,
            data_list: Union[Sequence[float], Sequence[Sequence[float]]],
            labels: Optional[Sequence[str]] = None,
            colors: Optional[Sequence[str]] = None,
            linestyles: Optional[Sequence[str]] = None,
            linewidth: float = 1.4,
            height: float = 2.2,
            title: Optional[str] = None,
            xlabel: Optional[str] = None,
            ylabel: str = 'CDF',
            legend_loc: str = 'best') -> Tuple[plt.Figure, plt.Axes]:

        if isinstance(data_list[0], (int, float, np.floating)):
            data_list = [data_list]
        n_series = len(data_list)

        if labels is None:
            labels = [f'S{i+1}' for i in range(n_series)]
        if colors is None:
            colors = [self._get_color(i) for i in range(n_series)]
        if linestyles is None:
            linestyles = [self._get_linestyle(i) for i in range(n_series)]

        fig, ax = self.figure(height=height)

        for i, (data, lab) in enumerate(zip(data_list, labels)):
            d = np.asarray(data, dtype=float)
            d = d[np.isfinite(d)]
            if d.size == 0:
                continue
            d_sorted = np.sort(d)
            y = np.arange(1, d_sorted.size + 1) / d_sorted.size
            color_i = colors[i] if self.color_mode not in ('texture-only', 'bw') else self.GRAYS[i % len(self.GRAYS)]
            ax.step(d_sorted, y, where='post', label=lab,
                    color=color_i, linestyle=linestyles[i], linewidth=linewidth, zorder=3)

        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)
        if title: ax.set_title(title, pad=4)
        if labels:
            ax.legend(frameon=False, loc=legend_loc)
        plt.tight_layout()
        return fig, ax

    # ---------- 便捷保存 ----------
    @staticmethod
    def save(fig: plt.Figure, path: str, bbox_inches: str = 'tight', dpi: Optional[int] = None):
        fig.savefig(path, bbox_inches=bbox_inches, dpi=dpi)




import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import List, Sequence, Optional, Tuple, Union

class ACMPlotter:
    """
    ACM 风格画图工具
    - 默认单栏宽 3.4 in（≈8.6 cm），Times 系 9pt，y 向虚线网格，向量输出友好
    - 自动配色（黑白适配，简单线条与标记）
    - color_mode:
        'auto'           -> 彩色 + 纹理（柱状图），折线/散点用颜色+标记/线型
        'bw'             -> 黑白适配，柱状图无纹理，折线/散点用黑白线条与标记
    """
    GRAYS = ['#000000', '#555555', '#888888', '#BBBBBB', '#DDDDDD']  # 适合黑白打印
    MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
    LINESTYLES = ['-', '--', '-.', ':']

    COL_WIDTHS_IN = {'single': 3.4, 'onehalf': 5.0, 'double': 7.16}

    def __init__(self,
                 base_size: int = 9,
                 font_family: Tuple[str, str] = ('Times New Roman', 'DejaVu Serif'),
                 column: str = 'single',
                 color_mode: str = 'auto'):
        self.base_size = base_size
        self.font_family = font_family
        self.column = column if column in self.COL_WIDTHS_IN else 'single'
        self.color_mode = color_mode
        self._apply_rc()

    def _apply_rc(self):
        mpl.rcParams.update({
            'font.size': self.base_size,
            'font.family': 'serif',
            'font.serif': list(self.font_family),
            'axes.titlesize': self.base_size,
            'axes.labelsize': self.base_size,
            'xtick.labelsize': max(self.base_size - 1, 6),
            'ytick.labelsize': max(self.base_size - 1, 6),
            'legend.fontsize': max(self.base_size - 1, 6),
            'axes.linewidth': 0.8,
            'grid.linestyle': '--',
            'grid.linewidth': 0.5,
            'grid.alpha': 0.5,
            'axes.grid': True,
            'axes.grid.axis': 'y',
            'savefig.dpi': 300,
            'figure.dpi': 150,
            'hatch.linewidth': 0.6,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'axes.unicode_minus': False,
        })

    # ---------- 基础：新建画布 ----------
    def figure(self, height: float = 2.2, width: Optional[float] = None) -> Tuple[plt.Figure, plt.Axes]:
        if width is None:
            width = self.COL_WIDTHS_IN.get(self.column, 3.4)
        fig, ax = plt.subplots(figsize=(width, height))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_axisbelow(True)
        ax.margins(x=0.02)
        return fig, ax

    # ---------- 自动样式选择 ----------
    def _get_color(self, i: int) -> str:
        return self.GRAYS[i % len(self.GRAYS)]

    def _get_marker(self, i: int) -> str:
        return self.MARKERS[i % len(self.MARKERS)]

    def _get_linestyle(self, i: int) -> str:
        return self.LINESTYLES[i % len(self.LINESTYLES)]

    # ---------- 柱状图（支持分组/误差线/自动纹理与配色） ----------
    def bar(self,
            categories: Sequence[Union[str, int]],
            series_list: Union[Sequence[float], Sequence[Sequence[float]]],
            labels: Optional[Sequence[str]] = None,
            yerr: Optional[Union[Sequence[float], Sequence[Sequence[float]]]] = None,
            show_values: bool = False,
            value_fmt: str = '%.0f',
            total_width: float = 0.8,
            edgecolor: str = 'black',
            linewidth: float = 0.8,
            colors: Optional[Sequence[str]] = None,
            hatches: Optional[Sequence[str]] = None,
            height: float = 2.2,
            title: Optional[str] = None,
            xlabel: Optional[str] = None,
            ylabel: Optional[str] = None,
            legend_loc: str = 'best') -> Tuple[plt.Figure, plt.Axes]:

        if isinstance(series_list[0], (int, float, np.floating)):
            series_list = [series_list]
        series_list = [np.asarray(s, dtype=float) for s in series_list]
        n_series = len(series_list)
        n_cats = len(categories)
        assert all(len(s) == n_cats for s in series_list), "每个系列长度需等于类别数"

        if labels is None:
            labels = [f'S{i+1}' for i in range(n_series)]

        if colors is None:
            colors = [self._get_color(i) for i in range(n_series)]
        if hatches is None:
            hatches = [None] * n_series

        fig, ax = self.figure(height=height)
        x = np.arange(n_cats)
        bar_width = total_width / max(n_series, 1)
        offsets = (np.arange(n_series) - (n_series - 1) / 2.0) * bar_width

        if yerr is not None:
            if isinstance(yerr[0], (int, float, np.floating)):
                yerr = [yerr] * n_series

        containers = []
        for i, (vals, lab) in enumerate(zip(series_list, labels)):
            err_i = None if yerr is None else yerr[i]
            bars = ax.bar(x + offsets[i], vals,
                          width=bar_width,
                          label=lab,
                          color=colors[i],
                          edgecolor=edgecolor,
                          linewidth=linewidth,
                          hatch=hatches[i],
                          yerr=err_i,
                          capsize=2 if err_i is not None else 0)
            containers.append(bars)
            if show_values:
                ax.bar_label(bars, fmt=value_fmt, padding=2, fontsize=mpl.rcParams['xtick.labelsize'])

        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)
        if title: ax.set_title(title, pad=4)
        if labels:
            ax.legend(frameon=False, handlelength=1.6, handletextpad=0.4, loc=legend_loc)

        plt.tight_layout()
        return fig, ax

    # ---------- 折线图（自动颜色/线型/标记） ----------
    def line(self,
             x: Sequence[float],
             y_list: Union[Sequence[float], Sequence[Sequence[float]]],
             labels: Optional[Sequence[str]] = None,
             colors: Optional[Sequence[str]] = None,
             linestyles: Optional[Sequence[str]] = None,
             markers: Optional[Sequence[str]] = None,
             linewidth: float = 1.4,
             markersize: float = 3.5,
             height: float = 2.2,
             title: Optional[str] = None,
             xlabel: Optional[str] = None,
             ylabel: Optional[str] = None,
             legend_loc: str = 'best') -> Tuple[plt.Figure, plt.Axes]:

        if isinstance(y_list[0], (int, float, np.floating)):
            y_list = [y_list]
        y_list = [np.asarray(y, dtype=float) for y in y_list]
        n_series = len(y_list)
        x = np.asarray(x, dtype=float)

        if labels is None:
            labels = [f'S{i+1}' for i in range(n_series)]
        if colors is None:
            colors = [self._get_color(i) for i in range(n_series)]
        if linestyles is None:
            linestyles = [self._get_linestyle(i) for i in range(n_series)]
        if markers is None:
            markers = [self._get_marker(i) for i in range(n_series)]

        fig, ax = self.figure(height=height)
        for i, (y, lab) in enumerate(zip(y_list, labels)):
            ax.plot(x, y,
                    label=lab,
                    color=colors[i],
                    linestyle=linestyles[i],
                    marker=markers[i],
                    markersize=markersize,
                    linewidth=linewidth)

        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)
        if title: ax.set_title(title, pad=4)
        if labels:
            ax.legend(frameon=False, loc=legend_loc)
        plt.tight_layout()
        return fig, ax

    # ---------- 散点图（带拟合线） ----------
    def scatter(self,
                x: Sequence[float],
                y: Sequence[float],
                label: Optional[str] = None,
                color: Optional[str] = None,
                marker: Optional[str] = None,
                s: float = 20,
                alpha: float = 0.9,
                fit: Optional[Union[str, int]] = None,
                fit_color: Optional[str] = None,
                fit_linestyle: str = '--',
                fit_linewidth: float = 1.2,
                height: float = 2.0,
                title: Optional[str] = None,
                xlabel: Optional[str] = None,
                ylabel: Optional[str] = None,
                legend_loc: str = 'best') -> Tuple[plt.Figure, plt.Axes]:

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        fig, ax = self.figure(height=height)

        c = color or self._get_color(0)
        m = marker or self._get_marker(0)

        sc = ax.scatter(x, y, s=s, label=label, color=c, marker=m, alpha=alpha)

        if fit is not None:
            deg = 1 if fit == 'linear' else int(fit)
            if len(x) >= deg + 1:
                coeffs = np.polyfit(x, y, deg)
                poly = np.poly1d(coeffs)
                xs = np.linspace(np.min(x), np.max(x), 200)
                ys = poly(xs)
                fc = fit_color or c
                ax.plot(xs, ys, color=fc, linestyle=fit_linestyle, linewidth=fit_linewidth,
                        label=f'Fit (deg={deg})' if label is None else f'{label} Fit')

        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)
        if title: ax.set_title(title, pad=4)
        if label or fit is not None:
            ax.legend(frameon=False, loc=legend_loc)
        plt.tight_layout()
        return fig, ax

    # ---------- CDF（经验分布函数） ----------
    def cdf(self,
            data_list: Union[Sequence[float], Sequence[Sequence[float]]],
            labels: Optional[Sequence[str]] = None,
            colors: Optional[Sequence[str]] = None,
            linestyles: Optional[Sequence[str]] = None,
            linewidth: float = 1.4,
            height: float = 2.2,
            title: Optional[str] = None,
            xlabel: Optional[str] = None,
            ylabel: str = 'CDF',
            legend_loc: str = 'best') -> Tuple[plt.Figure, plt.Axes]:

        if isinstance(data_list[0], (int, float, np.floating)):
            data_list = [data_list]
        n_series = len(data_list)

        if labels is None:
            labels = [f'S{i+1}' for i in range(n_series)]
        if colors is None:
            colors = [self._get_color(i) for i in range(n_series)]
        if linestyles is None:
            linestyles = [self._get_linestyle(i) for i in range(n_series)]

        fig, ax = self.figure(height=height)

        for i, (data, lab) in enumerate(zip(data_list, labels)):
            d = np.asarray(data, dtype=float)
            d_sorted = np.sort(d)
            y = np.arange(1, d_sorted.size + 1) / d_sorted.size
            ax.step(d_sorted, y, where='post', label=lab,
                    color=colors[i], linestyle=linestyles[i], linewidth=linewidth)

        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)
        if title: ax.set_title(title, pad=4)
        if labels:
            ax.legend(frameon=False, loc=legend_loc)
        plt.tight_layout()
        return fig, ax

    # ---------- 便捷保存 ----------
    @staticmethod
    def save(fig: plt.Figure, path: str, bbox_inches: str = 'tight', dpi: Optional[int] = None):
        fig.savefig(path, bbox_inches=bbox_inches, dpi=dpi)


# ---------------- 使用示例 ----------------
if __name__ == '__main__':
    plotter = IEEEPlotter(base_size=8, column='single', color_mode='auto')

    # 1) 柱状图（分组，自动颜色+纹理；含误差线与数值标签）
    cats = ['A','B','C','D']
    s1 = [12, 15, 13, 10]
    s2 = [10, 14, 12, 9]
    s3 = [8, 11, 9, 7]
    yerr = [ [1.0, 0.8, 1.1, 0.7],
             [0.9, 0.7, 1.0, 0.6],
             [0.8, 0.9, 0.7, 0.5] ]
    fig, ax = plotter.bar(
        categories=cats,
        series_list=[s1, s2, s3],
        labels=['Method 1', 'Method 2', 'Method 3'],
        yerr=None,
        show_values=False,
        ylabel='Score', xlabel='Category',
        title='Grouped Bar (IEEE Style)'
    )
    plotter.save(fig, 'ieee_bar.pdf')

    # 2) 折线图（自动颜色/线型/标记）
    x = [1,2,3,4]
    fig, ax = plotter.line(
        x=x,
        y_list=[ [0.7, 0.8, 0.83, 0.9], [0.6, 0.72, 0.78, 0.85] ],
        labels=['Alg A', 'Alg B'],
        xlabel='Epoch', ylabel='Accuracy',
        title='Line Plot'
    )
    plotter.save(fig, 'ieee_line.pdf')

    # 3) 散点图（带线性拟合）
    rng = np.random.default_rng(0)
    xs = np.linspace(0, 10, 30)
    ys = 2.0 * xs + 5 + rng.normal(0, 3, size=xs.size)
    fig, ax = plotter.scatter(xs, ys, label='Samples', fit='linear',
                              xlabel='X', ylabel='Y', title='Scatter with Fit')
    plotter.save(fig, 'ieee_scatter.pdf')

    # 4) CDF（经验分布）
    d1 = rng.normal(0, 1, 500)
    d2 = rng.normal(0.5, 1.2, 500)
    fig, ax = plotter.cdf([d1, d2], labels=['Dist A', 'Dist B'],
                          xlabel='Value', title='Empirical CDF')
    plotter.save(fig, 'ieee_cdf.pdf')
    
    
    plotter = ACMPlotter(base_size=9, column='single', color_mode='bw')

    # 示例：柱状图、折线图、散点图、CDF图
    cats = ['A', 'B', 'C', 'D']
    s1 = [12, 15, 13, 10]
    s2 = [10, 14, 12, 9]
    fig, ax = plotter.bar(cats, [s1, s2], labels=['Method 1', 'Method 2'], ylabel='Score', xlabel='Category')
    plotter.save(fig, 'acm_bar.pdf')

    x = [1, 2, 3, 4]
    fig, ax = plotter.line(x, [[0.7, 0.8, 0.83, 0.9], [0.6, 0.72, 0.78, 0.85]], labels=['Alg A', 'Alg B'])
    plotter.save(fig, 'acm_line.pdf')

    rng = np.random.default_rng(0)
    xs = np.linspace(0, 10, 30)
    ys = 2.0 * xs + 5 + rng.normal(0, 3, size=xs.size)
    fig, ax = plotter.scatter(xs, ys, fit='linear', xlabel='X', ylabel='Y')
    plotter.save(fig, 'acm_scatter.pdf')

    d1 = rng.normal(0, 1, 500)
    d2 = rng.normal(0.5, 1.2, 500)
    fig, ax = plotter.cdf([d1, d2], labels=['Dist A', 'Dist B'], xlabel='Value', title='Empirical CDF')
    plotter.save(fig, 'acm_cdf.pdf')
    
    
    
    