import {test} from 'node:test';
import assert from 'node:assert/strict';
import {issueSession,validSession,equalSecret} from '../lib/tokens.ts';
import {runInputSchema} from '../lib/contracts.ts';
import {boundedJSON} from '../lib/body.ts';

test('owner sessions reject tampering, expiry and missing configuration',()=>{
 const secret='a-private-test-secret';const now=1_000_000;const token=issueSession(secret,now);
 assert.equal(validSession(token,secret,now),true);
 assert.equal(validSession(token+'x',secret,now),false);
 assert.equal(validSession(token,'wrong',now),false);
 assert.equal(validSession(token,secret,now+8*86400000),false);
 assert.equal(validSession(token,undefined,now),false);
 assert.equal(equalSecret('short','longer'),false);
});
test('bounded request reader rejects oversized streamed data',async()=>{
 const request=new Request('https://miso.example',{method:'POST',body:JSON.stringify({value:'x'.repeat(1000)})});
 await assert.rejects(()=>boundedJSON(request,100));
});
test('provider identity and input contracts fail before any paid call',()=>{
 const text={provider:'jev',label:'test',payload:{state:'hello',questions:{q:{type:'noul',instructions:'Is this a greeting?'}}}};
 assert.equal(runInputSchema.safeParse(text).success,true);
 assert.equal(runInputSchema.safeParse({...text,payload:{...text.payload,model:'unapproved'}}).success,false);
 assert.equal(runInputSchema.safeParse({...text,payload:{...text.payload,state:'x'.repeat(12001)}}).success,false);
 assert.equal(runInputSchema.safeParse({...text,provider:'miso'}).success,false);
 const block={type:'image',source:{type:'base64',media_type:'image/png',data:'aGVsbG8='}};
 assert.equal(runInputSchema.safeParse({provider:'miso',label:'test',payload:{model:'mmso-joint-v3',input:[block,block],questions:{q:{type:'noul',question:'is the spoken command on the screen'}}}}).success,false);
});

import {sameOrigin} from '../lib/origin.ts';
test('origin validation handles Next localhost normalization and rejects foreign origins',()=>{
 const request=new Request('http://localhost:3047/api/session',{headers:{host:'127.0.0.1:3047',origin:'http://127.0.0.1:3047'}});
 assert.equal(sameOrigin(request),true);
 assert.equal(sameOrigin(new Request('https://miso.example/api',{headers:{host:'miso.example',origin:'https://evil.example'}})),false);
 assert.equal(sameOrigin(new Request('https://miso.example/api')),false);
});
