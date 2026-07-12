let selected='MM', layer=null, cards=[];
const $=id=>document.getElementById(id), pct=v=>`${(v*100).toFixed(1)}%`;
const colors={low:'#35b978',moderate:'#f3c846',high:'#f58b35',critical:'#e94455'};
const map = L.map('map', {
    zoomControl: false
}).setView([8, 108], 4);

L.control.zoom({
    position: 'bottomright'
}).addTo(map);

const satellite = L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    {
        attribution: 'Tiles &copy; Esri',
        maxZoom: 19
    }
);

const labels = L.tileLayer(
    'https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
    {
        attribution: 'Labels &copy; Esri'
    }
);

satellite.addTo(map);
labels.addTo(map);function toast(t){$('toast').textContent=t;$('toast').classList.add('show');setTimeout(()=>$('toast').classList.remove('show'),2300)}
async function api(url,opt){const r=await fetch(url,opt),b=await r.json().catch(()=>({detail:r.statusText}));if(!r.ok)throw new Error(typeof b.detail==='string'?b.detail:JSON.stringify(b.detail));return b}
function spark(data){const w=360,h=100,pts=data.map((d,i)=>`${i*w/(data.length-1)},${h-d.risk*85-7}`).join(' ');$('spark').innerHTML=`<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#21c99a" stop-opacity=".34"/><stop offset="1" stop-color="#21c99a" stop-opacity="0"/></linearGradient></defs><polygon points="0,100 ${pts} 360,100" fill="url(#g)"/><polyline points="${pts}" fill="none" stroke="#12a77f" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>`}
function countryCards(){ $('countries').innerHTML=cards.map(c=>`<div class="country ${c.code===selected?'active':''}" data-code="${c.code}"><div class="country-head"><b>${c.code}</b><i style="background:${colors[c.level]}"></i></div><strong>${pct(c.probability)}</strong><small>${c.name}</small></div>`).join('');document.querySelectorAll('.country').forEach(x=>x.onclick=()=>select(x.dataset.code)) }
async function select(code){selected=code;countryCards();const d=await api(`/api/v3/countries/${code}`);$('countryName').textContent=d.name;$('countryCode').textContent=d.code;$('prob').textContent=pct(d.probability);$('conf').textContent=pct(d.confidence);spark(d.timeline);$('sources').innerHTML=d.sources.map(s=>`<div class="source"><i></i><b>${s.name}</b><span>${s.age}</span><em>${Math.round(s.quality*100)}%</em></div>`).join('');$('population').textContent=d.impact.population_demo.toLocaleString();$('roads').textContent=`${d.impact.roads_km_demo} km`;$('health').textContent=d.impact.health_sites_demo;$('network').textContent=pct(d.impact.network_availability_demo);if(layer)map.removeLayer(layer);layer=L.geoJSON(d.polygon,{style:{color:colors[d.level],fillColor:colors[d.level],fillOpacity:.42,weight:3}}).addTo(map);map.flyTo(d.center,6,{duration:.65})}
async function load(){try{const r=await api('/api/v3/region');cards=r.countries;$('countryCount').textContent=r.summary.countries_monitored;$('highCount').textContent=r.summary.high_or_critical;$('sourceCount').textContent=r.summary.active_sources;$('reviewCount').textContent=r.summary.pending_reviews;const avg=cards.reduce((a,c)=>a+c.probability,0)/cards.length;$('regionRisk').textContent=pct(avg);countryCards();await select(selected)}catch(e){toast(e.message)}}
$('refresh').onclick=()=>{load();toast('Regional intelligence refreshed')};$('reviewAlert').onclick=()=>{$('modalTitle').textContent=`${selected} — Review CAP test alert`;$('result').textContent='';$('modal').classList.remove('hidden')};$('close').onclick=()=> $('modal').classList.add('hidden');async function review(decision){try{const body={decision,reviewer:$('reviewer').value,reason:$('reason').value,token:$('token').value};const r=await api(`/api/v3/alerts/${selected}/review`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});$('result').textContent=JSON.stringify(r,null,2);toast('Test workflow recorded')}catch(e){$('result').textContent=e.message;toast(e.message)}}$('approve').onclick=()=>review('approve');$('reject').onclick=()=>review('reject');setInterval(()=>$('clock').textContent=new Date().toLocaleString(),1000);load();