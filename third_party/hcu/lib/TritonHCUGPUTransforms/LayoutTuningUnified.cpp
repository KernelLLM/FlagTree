// Native equivalent of the frozen research175 + CTA ownership extractors.
// IDs only join facts from one normalized clone. No Python, seed, price,
// hardware resource measurements, timings or workload identities enter values.
#include "TritonHCUGPUTransforms/LayoutTuningUnified.h"
#include "TritonHCUGPUTransforms/LayoutTuningSSA.h"
#include "TritonHCUGPUTransforms/LayoutTuningFacts.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <vector>

namespace layoutdp {
namespace {
using Object = llvm::json::Object;
using Array = llvm::json::Array;
using Value = llvm::json::Value;
using IDs = std::vector<int64_t>;
using Coordinate = std::vector<int64_t>;

// Error state is sticky. Dummy intermediate values never reach the scorer:
// any missing/invalid input invalidates the complete candidate without fallback.
struct Reader {
  std::string error;
  bool check(bool condition, const char *why) {
    if (!condition && error.empty()) error = why;
    return condition;
  }
  const Object &object(const Value &v) {
    static const Object empty;
    auto *p = v.getAsObject();
    check(bool(p), "expected native fact object"); return p ? *p : empty;
  }
  const Object &object(const Object &o, const char *key) {
    static const Object empty;
    auto *p = o.getObject(key);
    check(bool(p), "required native fact object missing"); return p ? *p : empty;
  }
  const Array &array(const Object &o, const char *key) {
    static const Array empty;
    auto *p = o.getArray(key);
    check(bool(p), "required native fact array missing"); return p ? *p : empty;
  }
  std::string string(const Object &o, const char *key) {
    auto s = o.getString(key); check(bool(s), "required native fact string missing");
    return s.value_or("").str();
  }
  double number(const Object &o, const char *key) {
    auto n = o.getNumber(key);
    check(n && std::isfinite(*n) && *n >= 0, "required native number missing, negative or nonfinite");
    return n.value_or(0.);
  }
  int64_t integer(const Object &o, const char *key) {
    auto n = o.getNumber(key);
    bool valid=n && std::isfinite(*n) && *n>=0 && *n<double(std::numeric_limits<int64_t>::max()) && std::floor(*n)==*n;
    check(valid, "required native integer missing or invalid");
    return valid ? int64_t(*n) : 0;
  }
  IDs ids(const Object &o, const char *key) {
    IDs out;
    for (const auto &v : array(o,key)) {
      auto n=v.getAsInteger(); check(n && *n >= 0, "invalid native ID/extent");
      out.push_back(n.value_or(0));
    }
    return out;
  }
  bool flag(const Object &o, const char *key, bool expected = true) {
    auto b=o.getBoolean(key);
    return check(b && *b == expected, "required native availability/provenance flag invalid");
  }
};

std::string communication(const std::string &name) {
  if (name=="ttg.convert_layout") return "conversion";
  if (name=="tt.reduce" || name=="tt.scan") return "reduction";
  return name=="tt.dot" ? "dot" : "";
}
bool arithmetic(const std::string &name) {
  return name.compare(0,6,"arith.")==0 || name.compare(0,5,"math.")==0 || name=="tt.fp_to_fp";
}
bool oneOf(const std::string &name, std::initializer_list<const char *> choices) {
  for (const auto *c:choices) if (name==c) return true;
  return false;
}
struct Edge { int64_t source, target, operation; std::string kind; };
struct Sink { int64_t operation; std::string kind; };
struct Paths { bool terminal=false, recurrence=false; int64_t adds=0, arithmetic=0; };

class Extractor : public Reader {
  const Object &facts, &graph;
  std::map<std::string,double> totals;
  std::map<int64_t,const Object *> operations, blocks;
  std::vector<Edge> edges;
  std::map<int64_t,std::vector<Sink>> sinks;
  const Object *body(const Object &op, unsigned index) {
    const auto &regions=array(op,"regions");
    if (!check(index<regions.size(),"missing structured region")) return nullptr;
    auto regionBlocks=ids(object(regions[index]),"blocks");
    if (!check(regionBlocks.size()==1,"requires single-block structured region")) return nullptr;
    auto it=blocks.find(regionBlocks[0]);
    if (!check(it!=blocks.end(),"structured region references unknown block")) return nullptr;
    return it->second;
  }
  const Object *terminator(const Object &block) {
    auto list=ids(block,"operations");
    if (!check(!list.empty(),"structured block missing terminator")) return nullptr;
    auto it=operations.find(list.back());
    if (!check(it!=operations.end(),"unknown structured terminator")) return nullptr;
    return it->second;
  }
  void edge(int64_t a,int64_t b,const std::string &kind,int64_t op) { edges.push_back({a,b,op,kind}); }
  void sink(int64_t v,const std::string &kind,int64_t op) { sinks[v].push_back({op,kind}); }
  void term(const std::string &name,double value) {
    if (name.size()>=4 && name.substr(name.size()-4)=="_max") totals[name]=std::max(totals[name],value);
    else totals[name]+=value;
  }
  bool flow() {
    for (const auto &v:array(graph,"operations")) {
      const auto &op=object(v);auto id=integer(op,"operation");
      if (!check(operations.emplace(id,&op).second,"duplicate SSA operation ID")) return false;
    }
    for (const auto &v:array(graph,"blocks")) {
      const auto &b=object(v);auto id=integer(b,"block");
      if (!check(blocks.emplace(id,&b).second,"duplicate SSA block ID")) return false;
    }
    std::map<int64_t,const Object *> externs;
    if (const auto *rows=facts.getArray("extern_facts")) for (const auto &v:*rows) {
      const auto &row=object(v);
      if (!check(externs.emplace(integer(row,"operation"),&row).second,"duplicate extern semantic fact")) return false;
    }
    std::set<int64_t> conditions, matchedConditions;
    for (const auto &v:array(graph,"operations")) {
      const auto &op=object(v);auto id=integer(op,"operation");auto name=string(op,"name");
      auto operands=ids(op,"operands"),results=ids(op,"results");const auto &regions=array(op,"regions");
      if (name=="scf.condition") { conditions.insert(id);continue; }
      if (oneOf(name,{"builtin.module","tt.func","tt.return","scf.yield","tt.reduce.return","tt.scan.return",
          "arith.constant","tt.make_range","tt.get_program_id","tt.get_num_programs"})) continue;
      if (name=="scf.for") {
        auto *b=body(op,0);if (!b) return false;
        auto *end=terminator(*b);if (!end) return false;
        auto args=ids(*b,"arguments"),yielded=ids(*end,"operands");
        if (!check(operands.size()==3+results.size() && args.size()==1+results.size() &&
            string(*end,"name")=="scf.yield" && yielded.size()==results.size(),"invalid for value-port contract")) return false;
        for (unsigned i=0;i<3;++i) edge(operands[i],args[0],"loop-control",id);
        for (unsigned i=0;i<results.size();++i) {
          edge(operands[3+i],args[1+i],"loop-init",id);edge(operands[3+i],results[i],"loop-zero-trip",id);
          edge(yielded[i],args[1+i],"loop-recurrence",id);edge(yielded[i],results[i],"loop-exit",id);
          sink(yielded[i],"iteration-recurrence",id);
        }
      } else if (name=="scf.while") {
        if (!check(regions.size()==2,"invalid while region count")) return false;
        auto *before=body(op,0),*after=body(op,1);if (!before || !after) return false;
        auto *condition=terminator(*before),*yield=terminator(*after);if (!condition || !yield) return false;
        auto beforeArgs=ids(*before,"arguments"),afterArgs=ids(*after,"arguments");
        auto forwarded=ids(*condition,"operands"),yielded=ids(*yield,"operands");
        if (!check(string(*condition,"name")=="scf.condition" && string(*yield,"name")=="scf.yield" &&
            beforeArgs.size()==operands.size() && yielded.size()==operands.size() &&
            forwarded.size()==1+results.size() && afterArgs.size()==results.size(),
            "invalid while value-port contract")) return false;
        matchedConditions.insert(integer(*condition,"operation"));
        sink(forwarded[0],"conditional-control",id);
        for (unsigned i=0;i<operands.size();++i) {
          edge(operands[i],beforeArgs[i],"loop-init",id);
          edge(yielded[i],beforeArgs[i],"loop-recurrence",id);
          sink(yielded[i],"iteration-recurrence",id);
        }
        for (unsigned i=0;i<results.size();++i) {
          // Before executes even on a zero-trip while; the condition forwards
          // its own values to both the after region and the result ports.
          edge(forwarded[1+i],afterArgs[i],"loop-forward",id);
          edge(forwarded[1+i],results[i],"loop-exit",id);
          edge(forwarded[0],afterArgs[i],"conditional-control",id);
        }
      } else if (name=="scf.if") {
        if (!check(operands.size()==1,"invalid if condition")) return false;
        for (auto r:results) edge(operands[0],r,"conditional-control",id);
        for (unsigned i=0;i<regions.size();++i) {
          if (array(object(regions[i]),"blocks").empty() && results.empty()) continue;
          auto *b=body(op,i);if (!b) return false;
          auto *end=terminator(*b);if (!end) return false;
          auto yielded=ids(*end,"operands");
          if (!check(string(*end,"name")=="scf.yield" && yielded.size()==results.size(),"invalid if yield")) return false;
          for (unsigned j=0;j<results.size();++j) edge(yielded[j],results[j],"branch-yield",id);
        }
      } else if (name=="tt.reduce" || name=="tt.scan") {
        auto *b=body(op,0);if (!b) return false;
        auto *end=terminator(*b);if (!end) return false;
        auto args=ids(*b,"arguments"),returned=ids(*end,"operands");unsigned count=operands.size();
        if (!check(count && results.size()==count && args.size()==2*count && returned.size()==count &&
            string(*end,"name")== (name=="tt.scan"?"tt.scan.return":"tt.reduce.return"),"invalid collective tuple contract")) return false;
        std::map<int64_t,const Object *> producer;
        for (auto oid:ids(*b,"operations")) {
          auto it=operations.find(oid);if (!check(it!=operations.end(),"unknown combiner operation")) return false;
          for (auto r:ids(*it->second,"results")) producer[r]=it->second;
        }
        std::function<std::set<unsigned>(int64_t,std::set<int64_t>&)> arguments;
        arguments=[&](int64_t value,std::set<int64_t> &seen) {
          std::set<unsigned> result;
          if (!seen.insert(value).second) return result;
          auto it=std::find(args.begin(),args.end(),value);
          if (it!=args.end()) { result.insert(unsigned(it-args.begin())%count);return result; }
          auto p=producer.find(value);
          if (p!=producer.end()) for (auto x:ids(*p->second,"operands")) {
            auto found=arguments(x,seen);result.insert(found.begin(),found.end());
          }
          return result;
        };
        for (unsigned j=0;j<count;++j) {
          std::set<int64_t> seen;
          for (auto i:arguments(returned[j],seen)) edge(operands[i],results[j],"reduction",id);
        }
        for (auto x:operands) sink(x,"reduction-input",id);
      } else if (name=="tt.clampf" || name=="tt.extern_elementwise") {
        bool known=regions.empty() && results.size()==1;
        if (name=="tt.clampf") known &= operands.size()==3;
        else {
          auto it=externs.find(id);known &= !operands.empty() && it!=externs.end();
          if (it!=externs.end()) known &= it->second->getBoolean("available").value_or(false) && it->second->getBoolean("pure").value_or(false);
        }
        if (!check(known,"unknown clamp/extern value semantics")) return false;
        for (auto x:operands) edge(x,results[0],"arithmetic",id);
      } else if (name=="cf.br" || name=="cf.cond_br") {
        if (!check(regions.empty() && results.empty() && operands.size()==unsigned(name=="cf.cond_br"),
                   "CFG successor value ports unavailable")) return false;
        if (!operands.empty()) sink(operands[0],"conditional-control",id);
      } else if (name=="tt.load") {
        if (!check(results.size()==1 && !operands.empty() && operands.size()<=3,"invalid load ports")) return false;
        const char *kinds[]={"memory-address","memory-mask","memory-fallback"};
        for (unsigned i=0;i<operands.size();++i) edge(operands[i],results[0],kinds[i],id);
      } else if (name=="tt.store") {
        if (!check(results.empty() && operands.size()>=2 && operands.size()<=3,"invalid store ports")) return false;
        sink(operands[1],"global-store-data",id);
      } else if (name=="tt.atomic_rmw" || name=="tt.atomic_cas") {
        unsigned minimum=name=="tt.atomic_rmw"?2:3;
        if (!check(results.size()==1 && operands.size()>=minimum,"invalid atomic ports")) return false;
        edge(operands[0],results[0],"memory-address",id);
        for (unsigned i=1;i<minimum;++i) sink(operands[i],"atomic-update-data",id);
        if (operands.size()>minimum) edge(operands[minimum],results[0],"memory-mask",id);
      } else {
        if (!check(regions.empty(),"unmodeled structured region")) return false;
        if (name=="tt.fp_to_fp" && !check(operands.size()==1 && results.size()==1,"invalid float conversion ports")) return false;
        bool view=oneOf(name,{"tt.bitcast","tt.addptr","tt.splat","tt.broadcast","tt.expand_dims","tt.reshape",
            "tt.trans","tt.join","tt.split","tt.ptr_to_int","tt.int_to_ptr"});
        auto comm=communication(name);
        if (!check(arithmetic(name) || view || !comm.empty(),"unmodeled SSA operation")) return false;
        std::string kind=!comm.empty()?comm:arithmetic(name)?"arithmetic":"view";
        for (auto x:operands) for (auto r:results) edge(x,r,kind,id);
        if (!comm.empty()) for (auto x:operands) sink(x,comm+"-input",id);
      }
    }
    return check(conditions==matchedConditions,"condition outside modeled while contract") && error.empty();
  }
  bool stages() {
    if (!flow()) return false;
    std::map<int64_t,std::vector<const Edge *>> users,predecessors;
    for (const auto &e:edges) if (!oneOf(e.kind,{"loop-control","conditional-control","loop-recurrence","memory-address","memory-mask","memory-fallback"})) {
      users[e.source].push_back(&e);predecessors[e.target].push_back(&e);
    }
    std::map<int64_t,int64_t> depths;std::set<int64_t> visiting;
    std::function<int64_t(int64_t)> depth=[&](int64_t value) {
      auto it=depths.find(value);if (it!=depths.end()) return it->second;
      if (!check(visiting.insert(value).second,"undeclared cycle in static value flow")) return int64_t(0);
      int64_t result=0;
      for (auto *e:predecessors[value]) result=std::max(result,depth(e->source)+int64_t(oneOf(e->kind,{"conversion","reduction","dot"})));
      visiting.erase(value);depths[value]=result;return result;
    };
    for (const auto &v:array(graph,"operations")) {
      const auto &op=object(v);auto comm=communication(string(op,"name"));if (comm.empty()) continue;
      auto origin=integer(op,"operation");std::map<int64_t,Paths> cache;std::set<int64_t> stack;
      std::function<Paths(int64_t)> paths=[&](int64_t value) {
        auto it=cache.find(value);if (it!=cache.end()) return it->second;
        Paths result;
        if (!check(stack.insert(value).second,"undeclared cycle in static communication path")) return result;
        for (const auto &sink:sinks[value]) if (sink.operation!=origin) {
          result.terminal=true;result.recurrence |= sink.kind=="iteration-recurrence";
        }
        for (auto *e:users[value]) {
          if (oneOf(e->kind,{"conversion","reduction","dot"})) continue;
          auto p=paths(e->target);if (!p.terminal) continue;
          auto oi=operations.find(e->operation);if (!check(oi!=operations.end(),"unknown value user")) break;
          auto name=string(*oi->second,"name");
          result.terminal=true;result.recurrence |= p.recurrence;
          result.adds=std::max(result.adds,p.adds+int64_t(name=="arith.addf"));
          result.arithmetic=std::max(result.arithmetic,p.arithmetic+int64_t(arithmetic(name) || name=="tt.clampf" || name=="tt.extern_elementwise"));
        }
        stack.erase(value);cache[value]=result;return result;
      };
      Paths downstream;int64_t upstream=0;
      for (auto r:ids(op,"results")) {
        auto p=paths(r);downstream.adds=std::max(downstream.adds,p.adds);
        downstream.arithmetic=std::max(downstream.arithmetic,p.arithmetic);downstream.recurrence |= p.recurrence;
      }
      for (auto x:ids(op,"operands")) upstream=std::max(upstream,depth(x));
      auto prefix="ssa_"+comm+"_";
      term(prefix+"sites",1);
      term(prefix+"downstream_addf_depth_sum",downstream.adds);term(prefix+"downstream_addf_depth_max",downstream.adds);
      term(prefix+"downstream_arithmetic_depth_sum",downstream.arithmetic);term(prefix+"downstream_arithmetic_depth_max",downstream.arithmetic);
      term(prefix+"upstream_comm_stages_max",upstream);
      term("ssa_communication_stage_depth_max",upstream+1);
      term("ssa_communication_sites_reaching_recurrence",downstream.recurrence);
    }
    return error.empty();
  }
  bool geometry() {
    std::map<int64_t,const Object *> memory;
    for (const auto &v:array(facts,"memory_access_facts")) {
      const auto &row=object(v);
      if (!check(memory.emplace(integer(row,"operation"),&row).second,"duplicate memory access fact")) return false;
    }
    std::set<int64_t> expected,observed;
    for (const auto &[id,op]:operations) if (oneOf(string(*op,"name"),{"tt.load","tt.store"})) expected.insert(id);
    for (auto id:expected) if (!check(memory.count(id),"missing memory access association")) return false;
    if (!check(memory.size()==expected.size(),"extra memory access association")) return false;
    const auto &base=object(facts,"values");
    auto warps=integer(base,"num_warps"),width=integer(base,"threads_per_warp");
    if (!check(warps>0 && width>0 && integer(base,"num_ctas")==1,"invalid CTA geometry")) return false;
    std::map<std::string,long double> coordinates;
    for (const auto &v:array(graph,"coordinate_mappings")) {
      const auto &row=object(v);auto id=integer(row,"operation");
      auto mi=memory.find(id);
      if (!check(observed.insert(id).second && expected.count(id) && mi!=memory.end(),"invalid coordinate/memory association")) return false;
      const auto &mem=*mi->second;
      if (!flag(row,"available") || !flag(mem,"available")) return false;
      auto load=row.getBoolean("load");
      if (!check(load && mem.getBoolean("load")==load,"mismatched load/store association")) return false;
      auto kind=string(row,"kind");if (kind=="scalar") continue;
      if (!check(kind=="tensor","unsupported coordinate kind")) return false;
      auto shape=ids(row,"shape"),contiguity=ids(row,"contiguity");
      if (!check(!shape.empty() && contiguity.size()==shape.size(),"missing actual-rank shape/AxisInfo")) return false;
      for (unsigned i=0;i<shape.size();++i) if (!check(shape[i]>0 && contiguity[i]>0,"invalid actual-rank extent/contiguity")) return false;
      const auto &mapping=object(row,"mapping");
      if (!check(array(mapping,"outputs").size()==shape.size(),"coordinate output rank mismatch")) return false;
      struct Part { uint64_t size;std::vector<Coordinate> bases; };
      std::map<std::string,Part> parts;
      for (const auto &item:array(mapping,"inputs")) {
        const auto &part=object(item);Part p{uint64_t(integer(part,"size")),{}};
        for (const auto &bv:array(part,"bases")) {
          const auto *b=bv.getAsArray();if (!check(b && b->size()==shape.size(),"invalid coordinate basis rank")) return false;
          Coordinate vector;
          for (const auto &x:*b) {
            auto n=x.getAsInteger();if (!check(n && *n>=0,"invalid coordinate basis")) return false;
            vector.push_back(*n);
          }
          p.bases.push_back(std::move(vector));
        }
        if (!check(p.bases.size()<63 && p.size==(uint64_t(1)<<p.bases.size()),"coordinate size/basis mismatch")) return false;
        if (!check(parts.emplace(string(part,"dimension"),std::move(p)).second,"duplicate input dimension")) return false;
      }
      if (!check(parts.size()==4 && parts.count("register") && parts.count("lane") && parts.count("warp") && parts.count("block"),"unsupported input dimensions")) return false;
      if (!check(parts["block"].size==1 && parts["warp"].size==uint64_t(warps) && parts["lane"].size==uint64_t(width),"launch and coordinate dimensions disagree")) return false;
      uint64_t points=1;
      for (const auto &[name,part]:parts) {
        if (!check(part.size<=1048576/points,"coordinate enumeration budget exceeded")) return false;
        points*=part.size;
      }
      double weight=number(mem,"execution_weight");std::string side=*load?"load":"store";
      for (const auto *owner:{"register","lane","warp"}) {
        const auto &part=parts[owner];long double sum=0,square=0;
        for (const auto &basis:part.bases) for (unsigned i=0;i<shape.size();++i) {
          long double jump=static_cast<long double>(basis[i])/shape[i];sum+=jump;square+=jump*jump;
        }
        auto prefix="coordinate_"+side+"_"+owner+"_";
        coordinates[prefix+"jump_sum"]+=static_cast<long double>(weight)*sum;
        coordinates[prefix+"jump_square_sum"]+=static_cast<long double>(weight)*square;
      }
      int64_t low=0;
      for (const auto &basis:parts["warp"].bases) {
        int active=-1;
        for (unsigned i=0;i<basis.size();++i) if (basis[i]) { if (active!=-1) { active=-2;break; } active=i; }
        if (active<0) break;
        auto x=basis[active];if (!(x>0 && x<contiguity[active] && !(x&(x-1)))) break;
        ++low;
      }
      term("ssa_"+side+"_warp_contiguous_low_bits_sum",low);
      std::map<std::string,std::vector<Coordinate>> contributions;
      for (const auto *owner:{"register","lane","warp"}) {
        const auto &part=parts[owner];auto &rows=contributions[owner];rows.resize(part.size,Coordinate(shape.size(),0));
        for (uint64_t n=0;n<part.size;++n) for (unsigned bit=0;bit<part.bases.size();++bit) if (n&(uint64_t(1)<<bit))
          for (unsigned i=0;i<shape.size();++i) rows[n][i]^=part.bases[bit][i];
      }
      std::set<Coordinate> all;std::set<std::set<Coordinate>> warpSets;
      for (const auto &w:contributions["warp"]) {
        std::set<Coordinate> warpSet;
        for (const auto &l:contributions["lane"]) for (const auto &r:contributions["register"]) {
          Coordinate c(shape.size());
          for (unsigned i=0;i<shape.size();++i) {
            c[i]=r[i]^l[i]^w[i];if (!check(c[i]<shape[i],"coordinate outside actual payload shape")) return false;
          }
          warpSet.insert(std::move(c));
        }
        all.insert(warpSet.begin(),warpSet.end());warpSets.insert(std::move(warpSet));
      }
      auto bits=integer(mem,"element_bits");if (!check(bits>0,"unknown storage bits")) return false;
      auto prefix="cta_"+side+"_";double bytes=(bits+7)/8;
      term(prefix+"unique_logical_byte_sites_per_cta",weight*all.size()*bytes);
      term(prefix+"assigned_byte_slots_per_cta",weight*points*bytes);
      term(prefix+"warp_instruction_template_groups",weight*warps*number(mem,"instructions"));
      term(prefix+"distinct_warp_mapping_sites",weight*warpSets.size());
    }
    if (!check(observed==expected,"incomplete global coordinate association")) return false;
    for (const auto &[name,value]:coordinates) totals[name]=double(value);
    return error.empty();
  }
public:
  Extractor(const Object &f):facts(f),graph(object(f,"ssa_graph")) {}
  bool run(Object &values) {
    for (unsigned i=137;i<183;++i) totals[layoutUnifiedFeatureNames()[i]]=0.;
    flag(facts,"complete");flag(facts,"projection_succeeded");
    flag(facts,"seed_reference_input",false);flag(facts,"prices_used",false);flag(facts,"timing_inputs_used",false);
    if (!check(array(facts,"issues").empty(),"incomplete projected native facts")) return false;
    flag(graph,"available");flag(graph,"ids_are_inference_features",false);flag(graph,"exported_from_facts_clone");
    if (!check(string(graph,"coordinate_axis_scope")=="actual-payload-rank","unknown coordinate axis scope") || !error.empty()) return false;
    if (!stages() || !geometry()) return false;
    const auto &transport=object(facts,"shared_transport");
    flag(transport,"available");
    if (!check(array(transport,"issues").empty() && string(transport,"schema")=="bounded-shared-transport-v204" &&
        string(transport,"pipeline")=="shared-prefix+cleanup+opt-dot+RLC+reduce-data-duplication+optimize-LDS","wrong/incomplete shared transport contract")) return false;
    for (const auto *side:{"read","write"}) for (const auto *unit:{"vector_groups","word_work"}) {
      auto name=std::string("transport_shared_")+side+"_"+unit;
      totals[name]=number(object(transport,"values"),name.c_str());
    }
    const auto &base=object(facts,"values");
    for (unsigned i=0;i<137;++i) {
      const auto &name=layoutUnifiedFeatureNames()[i];values[name]=number(base,name.c_str());
    }
    for (const auto &[name,value]:totals) {
      if (!check(std::isfinite(value) && value>=0,"invalid derived unified feature")) return false;
      values[name]=value;
    }
    return error.empty() && check(values.size()==183,"unified feature count mismatch");
  }
};
} // namespace

const std::vector<std::string> &layoutUnifiedFeatureNames() {
  static const std::vector<std::string> names={
    "atomic_acquire_sites",
    "atomic_address_segment_work",
    "atomic_bitwise_work",
    "atomic_cta_scope_sites",
    "atomic_exchange_work",
    "atomic_float_add_work",
    "atomic_gpu_scope_sites",
    "atomic_integer_arithmetic_work",
    "atomic_release_sites",
    "atomic_result_block_sync_sites",
    "atomic_result_lds_load_words",
    "atomic_result_lds_store_words",
    "atomic_return_word_work",
    "atomic_scalar_bytes",
    "atomic_scalar_return_sites",
    "atomic_scalar_sites",
    "atomic_sites",
    "atomic_system_scope_sites",
    "atomic_tensor_bytes",
    "atomic_tensor_element_work",
    "atomic_tensor_word_work",
    "cfg_backedge_sites",
    "cfg_conditional_sites",
    "cfg_unconditional_sites",
    "collective_ballot_sites",
    "collective_bitwise_word_work",
    "collective_block_sync_sites",
    "collective_combine_word_work",
    "collective_lds_word_work",
    "collective_parallel_word_work",
    "collective_popcount_sites",
    "collective_serial_word_work",
    "collective_shuffle_word_work",
    "conditional_sites",
    "conversion_address_work",
    "conversion_declared_order_mismatch_sites",
    "conversion_declared_order_work",
    "conversion_temporary_words",
    "dot_multiply_accumulate_work",
    "dot_sites",
    "dynamic_loop_sites",
    "execution_branch_probabilities_known",
    "execution_context_present",
    "execution_cta_count",
    "execution_grid_known",
    "execution_grid_x",
    "execution_grid_y",
    "execution_grid_z",
    "execution_mask_population_known",
    "execution_resolved_loop_sites",
    "execution_resolved_trip_max",
    "execution_resolved_trip_sum",
    "execution_tensor_count",
    "execution_tensor_elements",
    "execution_tensor_logical_bytes",
    "execution_tensor_metadata_unknown",
    "execution_tensor_span_bytes",
    "execution_tensor_unit_stride_axes",
    "execution_unresolved_loop_sites",
    "explicit_block_sync_sites",
    "extern_constant_exponent_work",
    "extern_dynamic_exponent_work",
    "extern_exponent_magnitude_work",
    "extern_input_word_work",
    "extern_math_element_work",
    "extern_output_word_work",
    "extern_power_element_work",
    "index_coordinate_work",
    "index_dynamic_divrem_work",
    "index_regular_work",
    "lds_allocated_bytes",
    "lds_atomic_word_sites",
    "lds_block_sync_sites",
    "lds_load_word_sites",
    "lds_pointer_coordinate_sites",
    "lds_store_word_sites",
    "lds_vector_ops",
    "lds_warp_sync_sites",
    "load_element_bytes_sum",
    "load_instructions",
    "load_sites",
    "load_vector_elements_sum",
    "load_work",
    "masked_load_sites",
    "masked_store_sites",
    "num_ctas",
    "num_stages",
    "num_warps",
    "opaque_operation_sites",
    "payload_bitcast_sites",
    "payload_clamp_elements",
    "payload_divrem_work",
    "payload_float_convert_bf16_elements",
    "payload_float_convert_elements",
    "payload_float_convert_fp8_elements",
    "payload_float_convert_input_words",
    "payload_float_convert_output_words",
    "payload_float_rounding_sites",
    "payload_integer_convert_elements",
    "payload_regular_work",
    "payload_special_math_work",
    "post_dot_conversion_temporary_words",
    "post_dot_lds_allocated_bytes",
    "post_dot_lds_block_sync_sites",
    "post_dot_lds_load_word_sites",
    "post_dot_lds_store_word_sites",
    "post_dot_lds_vector_ops",
    "post_dot_lds_warp_sync_sites",
    "post_dot_shuffle_word_ops",
    "reduce_block_sync_sites",
    "reduce_combine_word_work",
    "reduce_lane_combine_work",
    "reduce_lds_load_words",
    "reduce_lds_store_words",
    "reduce_local_combine_work",
    "reduce_shuffle_word_work",
    "scalar_arithmetic_sites",
    "scalar_load_bytes",
    "scalar_load_sites",
    "scalar_store_bytes",
    "scalar_store_sites",
    "scan_block_sync_sites",
    "scan_carry_combine_work",
    "scan_lane_combine_work",
    "scan_lds_load_words",
    "scan_lds_store_words",
    "scan_local_combine_work",
    "scan_reverse_shuffle_word_work",
    "scan_shuffle_word_work",
    "shuffle_word_ops",
    "store_element_bytes_sum",
    "store_instructions",
    "store_sites",
    "store_vector_elements_sum",
    "store_work",
    "threads_per_warp",
    "unknown_memory_sites",
    "ssa_conversion_sites",
    "ssa_conversion_downstream_addf_depth_sum",
    "ssa_conversion_downstream_addf_depth_max",
    "ssa_conversion_downstream_arithmetic_depth_sum",
    "ssa_conversion_downstream_arithmetic_depth_max",
    "ssa_conversion_upstream_comm_stages_max",
    "ssa_reduction_sites",
    "ssa_reduction_downstream_addf_depth_sum",
    "ssa_reduction_downstream_addf_depth_max",
    "ssa_reduction_downstream_arithmetic_depth_sum",
    "ssa_reduction_downstream_arithmetic_depth_max",
    "ssa_reduction_upstream_comm_stages_max",
    "ssa_dot_sites",
    "ssa_dot_downstream_addf_depth_sum",
    "ssa_dot_downstream_addf_depth_max",
    "ssa_dot_downstream_arithmetic_depth_sum",
    "ssa_dot_downstream_arithmetic_depth_max",
    "ssa_dot_upstream_comm_stages_max",
    "ssa_communication_stage_depth_max",
    "ssa_communication_sites_reaching_recurrence",
    "ssa_load_warp_contiguous_low_bits_sum",
    "ssa_store_warp_contiguous_low_bits_sum",
    "transport_shared_read_vector_groups",
    "transport_shared_write_vector_groups",
    "transport_shared_read_word_work",
    "transport_shared_write_word_work",
    "coordinate_load_register_jump_sum",
    "coordinate_load_register_jump_square_sum",
    "coordinate_load_lane_jump_sum",
    "coordinate_load_lane_jump_square_sum",
    "coordinate_load_warp_jump_sum",
    "coordinate_load_warp_jump_square_sum",
    "coordinate_store_register_jump_sum",
    "coordinate_store_register_jump_square_sum",
    "coordinate_store_lane_jump_sum",
    "coordinate_store_lane_jump_square_sum",
    "coordinate_store_warp_jump_sum",
    "coordinate_store_warp_jump_square_sum",
    "cta_load_unique_logical_byte_sites_per_cta",
    "cta_load_assigned_byte_slots_per_cta",
    "cta_load_warp_instruction_template_groups",
    "cta_load_distinct_warp_mapping_sites",
    "cta_store_unique_logical_byte_sites_per_cta",
    "cta_store_assigned_byte_slots_per_cta",
    "cta_store_warp_instruction_template_groups",
    "cta_store_distinct_warp_mapping_sites",
  };
  return names;
}

Object enrichLayoutUnifiedFacts(Object facts,bool diagnostics) {
  Object values;Extractor extractor(facts);bool valid=extractor.run(values);
  facts["complete"]=valid;
  if (!valid) {
    facts["projection_succeeded"]=false;
    facts["issues"]=Array{extractor.error.empty()?"unified facts unavailable":extractor.error};
    values.clear();
  }
  facts["values"]=std::move(values);Array names;
  if (valid) for (const auto &name:layoutUnifiedFeatureNames()) names.push_back(name);
  facts["feature_names"]=std::move(names);
  facts["input_contract_sha256"]=layoutUnifiedContractSHA;
  facts["native_unified_extraction"]=Object{{"contract",layoutUnifiedContract},{"columns",183},
      {"derived_columns",46},{"removed_legacy_resource_columns",2},{"python_used",false},
      {"same_normalized_clone",true},{"unknown_values_imputed",false},
      {"runtime_active_warps_known",false},{"occupancy_known",false}};
  if (!diagnostics) for (const char *key:{"ssa_graph","resources","index_dag","memory_access_facts","extern_facts", "dot_operand_lookahead","shared_transport"}) facts.erase(key);
  return facts;
}

Object projectLayoutUnifiedFacts(mlir::ModuleOp candidate,const LayoutProjectionConfig &config,bool diagnostics) {
  return enrichLayoutUnifiedFacts(projectLayoutTransportFacts(candidate,config),diagnostics);
}
} // namespace layoutdp
