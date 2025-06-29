use std::sync::mpsc;
use rosc::OscPacket;

pub enum OscCommand {
    SetHeight(f32),
}

pub fn osc_thread_main(osc_sender: mpsc::Sender<OscCommand>) {
    log::info!("OSC Thread started");

    let address = "127.0.0.1:7000"; // Why does to_string dereference this?
    let socket = match std::net::UdpSocket::bind(address) {
        Ok(socket) => socket,
        Err(error) => {
            log::error!("Could not bind UDP socket for OSC: {}", error);
            return;
        }
    };

    log::info!("Listening on {}", address);

    let mut buf = [0u8; rosc::decoder::MTU];

    loop {
        match socket.recv_from(&mut buf) {
            Ok((size, addr)) => {
                println!("Received packet with size {} from: {}", size, addr);
                let (_, packet) = rosc::decoder::decode_udp(&buf[..size]).unwrap();
                match packet {
                    OscPacket::Message(msg) => {
                        println!("OSC address: {}", msg.addr);
                        println!("OSC arguments: {:?}", msg.args);
                    }
                    OscPacket::Bundle(bundle) => {
                        println!("OSC Bundle: {:?}", bundle);
                    }
                }
            }
            Err(e) => {
                println!("Error receiving from socket: {}", e);
                break;
            }
        }
    }
    log::info!("OSC Thread finished");
}