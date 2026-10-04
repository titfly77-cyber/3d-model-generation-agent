"""Explicitly labelled, locally modelled examples. Never represented as LLM output."""
import math


def robot():
    materials = {
        'sage':{'color':'#8caa69','metallic':.3,'roughness':.34},
        'dark_green':{'color':'#314b35','metallic':.35,'roughness':.38},
        'rubber':{'color':'#242726','roughness':.76},
        'alloy':{'color':'#b7bcb0','metallic':.8,'roughness':.27},
        'ivory':{'color':'#e8e5d4','roughness':.3},
        'glass':{'color':'#243e46','metallic':.65,'roughness':.18},
        'amber':{'color':'#ffb64b','emission':1.8,'roughness':.3},
        'blue':{'color':'#6fa9ba','metallic':.7,'roughness':.3},
    }
    parts=[]
    def add(name,kind,mat,pos=None,scale=None,**kw):
        p={'name':name,'type':kind,'material':mat,**kw}
        if pos is not None:p['position']=pos
        if scale is not None:p['scale']=scale
        parts.append(p)
    add('主承载底盘','box','dark_green',[0,0,.69],[1.82,2.15,.35],bevel=.12)
    add('一体式机身','box','sage',[0,0,1.05],[1.72,1.98,.65],bevel=.16)
    add('上壳分层','box','sage',[0,.10,1.37],[1.65,1.7,.14],bevel=.06)
    add('前传感器黑色面板','box','glass',[0,-.998,1.1],[1.32,.055,.31],bevel=.04)
    add('前保险杠','box','alloy',[0,-1.12,.65],[1.87,.17,.13],bevel=.05)
    add('后保险杠','box','alloy',[0,1.12,.65],[1.87,.17,.13],bevel=.05)
    for side in (-1,1):
        for y in (-.7,.7):
            label=f'{side}_{y}'
            add('车轮_'+label,'cylinder','rubber',[side*1.02,y,.44],radius=.43,depth=.29,rotation=[0,90,0],bevel=.05,segments=48)
            add('轮毂_'+label,'cylinder','alloy',[side*1.178,y,.44],radius=.26,depth=.035,rotation=[0,90,0],bevel=.02)
            add('轮芯_'+label,'cylinder','dark_green',[side*1.20,y,.44],radius=.15,depth=.035,rotation=[0,90,0],bevel=.015)
            add('轮轴_'+label,'beam','alloy',start=[side*.7,y,.44],end=[side*1.04,y,.44],radius=.07)
            add('悬挂_'+label,'beam','dark_green',start=[side*.77,y-.2,.77],end=[side*1.02,y,.44],radius=.055)
            for i in range(12):
                a=2*math.pi*i/12
                add('胎纹_'+label+'_'+str(i),'box','rubber',[side*1.02,y+.427*math.sin(a),.44+.427*math.cos(a)],[.27,.085,.038],rotation=[-math.degrees(a),0,0],bevel=.008)
            for i in range(4):
                a=math.pi*i/2
                add('轮毂螺栓_'+label+'_'+str(i),'sphere','ivory',[side*1.223,y+.2*math.sin(a),.44+.2*math.cos(a)],[.025,.025,.025])
        add('侧防撞条_'+str(side),'box','dark_green',[side*.876,0,1.04],[.07,1.0,.18],bevel=.02)
        add('前灯_'+str(side),'box','amber',[side*.64,-1.037,1.08],[.20,.04,.095],bevel=.02)
        add('后指示灯_'+str(side),'box','amber',[side*.60,1.006,.98],[.15,.04,.09],bevel=.02)
    add('顶部太阳能面板','box','glass',[0,.25,1.456],[1.3,1.13,.035],bevel=.012)
    for x in (-.48,-.24,0,.24,.48):
        add('光伏导线_'+str(x),'box','blue',[x,.25,1.478],[.009,1.04,.004],bevel=0)
    for y in (-.20,.05,.30,.55):
        add('光伏横线_'+str(y),'box','blue',[0,y,1.479],[1.20,.008,.004],bevel=0)
    add('传感器底座','cylinder','dark_green',[0,-.58,1.49],radius=.23,depth=.12)
    add('传感器支柱','cylinder','alloy',[0,-.58,1.69],radius=.055,depth=.35)
    add('感知头部','box','ivory',[0,-.60,1.96],[.62,.38,.35],bevel=.10)
    add('镜头面板','box','glass',[0,-.797,1.98],[.44,.025,.20],bevel=.06)
    for x in (-.12,.12):
        add('镜头_'+str(x),'cylinder','blue',[x,-.822,1.98],radius=.059,depth=.022,rotation=[90,0,0])
    add('顶部激光雷达','cylinder','dark_green',[0,-.60,2.20],radius=.14,depth=.11)
    add('雷达盖','cylinder','sage',[0,-.60,2.269],radius=.145,depth=.034)
    add('天线','beam','dark_green',start=[.59,.78,1.44],end=[.59,.78,1.94],radius=.017)
    add('天线端点','sphere','amber',[.59,.78,1.95],[.034,.034,.034])
    for i in range(5):
        add('后散热口_'+str(i),'box','dark_green',[-.36+i*.18,1.005,1.20],[.08,.018,.20],bevel=.018)
    return {'title':'TERRA · 农业巡检机器人','description':'低重心四轮底盘，鼠尾草绿模块化外壳，太阳能顶板与双目感知模组。','materials':materials,'parts':parts}


def lamp():
    return {'title':'ORBIT · 环形台灯','description':'暖白圆环灯体，黄铜支架与陶瓷底座，柔和的现代桌面照明。','materials':{
        'ceramic':{'color':'#d8d1bf','roughness':.33},'brass':{'color':'#b28a4b','metallic':.8,'roughness':.28},
        'light':{'color':'#fff1ca','emission':2},'dark':{'color':'#3b4038','roughness':.5}},'parts':[
        {'name':'底座','type':'lathe','material':'ceramic','points':[[0,0],[.55,0],[.62,.07],[.61,.16],[.52,.22],[.12,.25],[0,.25]],'bevel':.015},
        {'name':'底座金属边','type':'torus','material':'brass','position':[0,0,.09],'radius':.605,'tube':.015},
        {'name':'立柱','type':'cylinder','material':'brass','position':[0,0,.95],'radius':.035,'depth':1.45},
        {'name':'灯体','type':'torus','material':'ceramic','position':[0,0,2.05],'radius':.67,'tube':.095,'rotation':[90,0,0],'segments':64},
        {'name':'发光环','type':'torus','material':'light','position':[0,-.052,2.05],'radius':.67,'tube':.053,'rotation':[90,0,0],'segments':64},
        {'name':'背部环','type':'torus','material':'brass','position':[0,.06,2.05],'radius':.67,'tube':.035,'rotation':[90,0,0],'segments':64},
        {'name':'旋钮','type':'cylinder','material':'brass','position':[.3,-.23,.23],'radius':.075,'depth':.07},
        {'name':'开关','type':'cylinder','material':'dark','position':[.3,-.23,.271],'radius':.046,'depth':.015},
        {'name':'导线','type':'curve','material':'dark','points':[[0,.55,.1],[0,.8,.04],[.4,1.1,.02],[.65,1.3,.02]],'tube':.015}
    ]}


def chair():
    mats={'wood':{'color':'#876341','roughness':.4},'leather':{'color':'#b96136','roughness':.65},'metal':{'color':'#292c2c','metallic':.65,'roughness':.3}}
    parts=[{'name':'坐垫','type':'box','material':'leather','position':[0,0,.92],'scale':[1.30,1.22,.22],'bevel':.10},
           {'name':'靠背','type':'box','material':'leather','position':[0,.53,1.52],'scale':[1.30,.18,1.0],'rotation':[-12,0,0],'bevel':.085},
           {'name':'座框','type':'box','material':'wood','position':[0,0,.77],'scale':[1.37,1.27,.12],'bevel':.025}]
    for side in (-1,1):
        for y in (-.48,.48):
            parts.append({'name':f'椅腿{side}{y}','type':'beam','material':'wood','start':[side*.62,y*1.2,0],'end':[side*.52,y,.79],'radius':.045})
        parts.extend([
            {'name':f'扶手{side}','type':'box','material':'wood','position':[side*.78,.0,1.25],'scale':[.13,1.25,.10],'bevel':.035},
            {'name':f'扶手支撑{side}','type':'beam','material':'metal','start':[side*.65,-.39,.79],'end':[side*.78,-.39,1.23],'radius':.026},
            {'name':f'背部框架{side}','type':'beam','material':'wood','start':[side*.64,.50,.73],'end':[side*.64,.73,1.98],'radius':.04},
        ])
    return {'title':'FOLD · 休闲扶手椅','description':'胡桃木结构与赤陶色软垫，轻盈的支撑和温暖的材质对比。','materials':mats,'parts':parts}
